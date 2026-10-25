#!/usr/bin/env python3
"""
Two-stage cascaded U-Net for liver and tumor segmentation on the
"LiTS Dataset: Liver and Tumor Segmentation 256x256" Kaggle dataset
(sodko3/lits-dataset-liver-and-tumor-segmentation-256x256).

Stage 1  liver_unet.keras : CT slice                 -> liver mask
Stage 2  tumor_unet.keras : CT slice * liver mask    -> tumor mask

Both models are saved to  <out-dir>/liver_unet.keras  and  <out-dir>/tumor_unet.keras.

Usage
-----
Kaggle notebook (add the dataset, turn on a GPU, then run in a cell):
    !python train_liver_tumor_unet.py

Local machine:
    python train_liver_tumor_unet.py --data-root /path/to/dataset --out-dir models

If auto-detection of images/masks guesses wrong, point at the folders directly:
    python train_liver_tumor_unet.py --images-dir /path/images --masks-dir /path/masks

Quick smoke test (a few minutes):
    python train_liver_tumor_unet.py --max-slices 2000 --epochs-liver 2 --epochs-tumor 2

Inference with the saved models:
    python train_liver_tumor_unet.py --predict slice1.png slice2.png --out-dir models

Mask label convention (LiTS): 0 = background, 1 = liver, 2 = tumor. The script
accepts masks stored as 0/1/2 or scaled (e.g. 0/127/255 or 0/128/255) and
works out which one it is by itself.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
try:  # Keras 3 (TF >= 2.16)
    import keras  # noqa: E402
    from keras import layers  # noqa: E402
    if int(keras.__version__.split(".")[0]) < 3:
        raise ImportError
except ImportError:  # older TF with bundled Keras 2
    from tensorflow import keras  # noqa: E402
    from tensorflow.keras import layers  # noqa: E402

register = (keras.saving.register_keras_serializable if hasattr(keras, "saving")
            else keras.utils.register_keras_serializable)

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
IMG_EXTS = {".png", ".jpg", ".jpeg"}
DEFAULT_KAGGLE_SLUG = "lits-dataset-liver-and-tumor-segmentation-256x256"


# --------------------------------------------------------------------------- #
# Losses and metrics (registered so the .keras files reload cleanly)
# --------------------------------------------------------------------------- #
def _flat(t):
    return tf.reshape(tf.cast(t, tf.float32), [-1])


@register(package="liver_tumor")
def dice_coef(y_true, y_pred, smooth=1.0):
    yt, yp = _flat(y_true), _flat(y_pred)
    inter = tf.reduce_sum(yt * yp)
    return (2.0 * inter + smooth) / (tf.reduce_sum(yt) + tf.reduce_sum(yp) + smooth)


@register(package="liver_tumor")
def iou_coef(y_true, y_pred, smooth=1.0):
    yt, yp = _flat(y_true), _flat(y_pred)
    inter = tf.reduce_sum(yt * yp)
    union = tf.reduce_sum(yt) + tf.reduce_sum(yp) - inter
    return (inter + smooth) / (union + smooth)


def _bce(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred))


@register(package="liver_tumor")
def liver_loss(y_true, y_pred):
    """BCE + soft Dice. Liver is a large, easy target."""
    return _bce(y_true, y_pred) + (1.0 - dice_coef(y_true, y_pred))


@register(package="liver_tumor")
def tumor_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1.0):
    """BCE + Tversky. Tumors are small, so false negatives are penalised more."""
    yt, yp = _flat(y_true), _flat(y_pred)
    tp = tf.reduce_sum(yt * yp)
    fp = tf.reduce_sum((1.0 - yt) * yp)
    fn = tf.reduce_sum(yt * (1.0 - yp))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return _bce(y_true, y_pred) + (1.0 - tversky)


CUSTOM_OBJECTS = {
    "dice_coef": dice_coef,
    "iou_coef": iou_coef,
    "liver_loss": liver_loss,
    "tumor_loss": tumor_loss,
}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def conv_block(x, filters, dropout=0.0):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                          kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    if dropout:
        x = layers.SpatialDropout2D(dropout)(x)
    return x


def build_unet(input_shape=(256, 256, 1), base_filters=32, depth=4, name="unet"):
    inputs = layers.Input(input_shape)
    x, skips = inputs, []
    for d in range(depth):
        x = conv_block(x, base_filters * 2 ** d)
        skips.append(x)
        x = layers.MaxPooling2D()(x)
    x = conv_block(x, base_filters * 2 ** depth, dropout=0.2)
    for d in reversed(range(depth)):
        x = layers.Conv2DTranspose(base_filters * 2 ** d, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skips[d]])
        x = conv_block(x, base_filters * 2 ** d)
    outputs = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32")(x)
    return keras.Model(inputs, outputs, name=name)


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
# A path component counts as "mask-like" if it matches this.
MASK_WORDS = re.compile(
    r"segmentation|segment|annotation|annot|mask|label"
    r"|(?<![a-z])seg(?![a-z])|(?<![a-z])gt(?![a-z])"
)
# Words stripped from folder / file names so an image and its mask share a key.
STRIP_WORDS = re.compile(
    r"segmentations?|annotations?|volumes?|segments?|annots?|images?|masks?"
    r"|labels?|imgs?|scans?|seg|vol|ct|gt"
)


def _norm_token(s):
    s = STRIP_WORDS.sub("", s.lower())
    return re.sub(r"[-_.\s]+", "_", s).strip("_")


def norm_key(rel):
    parts = [_norm_token(p) for p in rel.parent.parts]
    parts = [p for p in parts if p]
    return "/".join(parts + [_norm_token(rel.stem)])


def is_mask_path(rel):
    comps = list(rel.parent.parts) + [rel.stem]
    return any(MASK_WORDS.search(c.lower()) for c in comps)


def list_images(root):
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTS)


def find_data_root(arg):
    if arg:
        return Path(arg)
    for cand in (f"/kaggle/input/{DEFAULT_KAGGLE_SLUG}", "/kaggle/input", "./dataset", "./data"):
        if Path(cand).exists():
            return Path(cand)
    sys.exit("Could not find the dataset. Pass --data-root (or --images-dir and --masks-dir).")


def discover_pairs(args):
    imgs, msks, dup = {}, {}, 0
    if args.images_dir and args.masks_dir:
        for root, store in ((Path(args.images_dir), imgs), (Path(args.masks_dir), msks)):
            for p in list_images(root):
                k = norm_key(p.relative_to(root))
                dup += k in store
                store[k] = p
        source = f"{args.images_dir}  +  {args.masks_dir}"
    else:
        root = find_data_root(args.data_root)
        for p in list_images(root):
            rel = p.relative_to(root)
            store = msks if is_mask_path(rel) else imgs
            k = norm_key(rel)
            dup += k in store
            store[k] = p
        source = str(root)

    keys = sorted(set(imgs) & set(msks))
    print(f"[data] source: {source}")
    print(f"[data] images found: {len(imgs):,} | masks found: {len(msks):,} | paired: {len(keys):,}")
    if dup:
        print(f"[data] WARNING: {dup} filename collisions while pairing; check the folder layout.")
    if not keys:
        print("[data] Sample image paths:", [str(p) for p in list(imgs.values())[:3]])
        print("[data] Sample mask paths :", [str(p) for p in list(msks.values())[:3]])
        sys.exit("No image/mask pairs found. Re-run with --images-dir and --masks-dir.")
    unpaired = len(imgs) - len(keys)
    if unpaired > 0.05 * len(imgs):
        print(f"[data] WARNING: {unpaired:,} images have no matching mask and will be ignored.")
    print(f"[data] example pair:\n         {imgs[keys[0]]}\n         {msks[keys[0]]}")
    return [(imgs[k], msks[k], k) for k in keys]


def patient_id(img_path, key):
    s = f"{img_path.parent.name}/{img_path.stem}".lower()
    m = re.search(r"(?:volume|vol|patient|case|scan|ct)[-_ ]?(\d+)", s)
    if m:
        return m.group(1)
    m = re.search(r"(\d+)", key)
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# Mask handling
# --------------------------------------------------------------------------- #
def detect_encoding(mask_paths, n=300):
    rng = random.Random(SEED)
    sample = rng.sample(mask_paths, min(n, len(mask_paths)))
    values = set()
    for p in sample:
        values |= set(np.unique(np.array(Image.open(p).convert("L"))).tolist())
    gmax = max(values)
    enc = "index" if gmax <= 2 else "scaled"
    print(f"[mask] unique pixel values in a {len(sample)}-mask sample: {sorted(values)[:12]}"
          f"{' ...' if len(values) > 12 else ''}  -> encoding: {enc}")
    if gmax <= 1:
        print("[mask] WARNING: only values {0,1} found. If this dataset has no tumor label "
              "(value 2), the tumor stage will have nothing to train on.")
    return enc


def to_label_np(m, enc):
    if enc == "index":
        return m.astype(np.uint8)
    return (m >= 64).astype(np.uint8) + (m >= 192).astype(np.uint8)


def scan_masks(mask_paths, enc, workers=16):
    def one(p):
        lbl = to_label_np(np.array(Image.open(p).convert("L")), enc)
        return int((lbl >= 1).sum()), int((lbl == 2).sum())

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, mask_paths))
    arr = np.array(res, dtype=np.int64)
    print(f"[scan] scanned {len(mask_paths):,} masks in {time.time() - t0:.0f}s | "
          f"slices with liver: {(arr[:, 0] > 0).sum():,} | with tumor: {(arr[:, 1] > 0).sum():,}")
    return arr[:, 0], arr[:, 1]


# --------------------------------------------------------------------------- #
# tf.data pipeline
# --------------------------------------------------------------------------- #
def make_loader(task, enc, size):
    def load(ip, mp):
        img = tf.io.decode_image(tf.io.read_file(ip), channels=1, expand_animations=False)
        img = tf.image.resize(img, [size, size], method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        m = tf.io.decode_image(tf.io.read_file(mp), channels=1, expand_animations=False)
        m = tf.image.resize(m, [size, size], method="nearest")
        m = tf.cast(m, tf.float32)
        if enc == "scaled":
            lbl = tf.cast(m >= 64, tf.float32) + tf.cast(m >= 192, tf.float32)
        else:
            lbl = m
        liver = tf.cast(lbl >= 1, tf.float32)
        tumor = tf.cast(tf.equal(lbl, 2), tf.float32)
        if task == "liver":
            return img, liver
        if task == "tumor":
            return img * liver, tumor      # tumor stage only sees the liver region
        return img, lbl                    # "raw": used for cascade evaluation

    return load


def augment(x, y):
    flip_h = tf.random.uniform([]) < 0.5
    flip_v = tf.random.uniform([]) < 0.3
    x = tf.cond(flip_h, lambda: tf.image.flip_left_right(x), lambda: x)
    y = tf.cond(flip_h, lambda: tf.image.flip_left_right(y), lambda: y)
    x = tf.cond(flip_v, lambda: tf.image.flip_up_down(x), lambda: x)
    y = tf.cond(flip_v, lambda: tf.image.flip_up_down(y), lambda: x)
    x = tf.clip_by_value(x * tf.random.uniform([], 0.9, 1.1), 0.0, 1.0)  # keeps zeros at zero
    return x, y


def make_dataset(img_paths, msk_paths, task, enc, size, batch, training):
    ds = tf.data.Dataset.from_tensor_slices(([str(p) for p in img_paths],
                                             [str(p) for p in msk_paths]))
    if training:
        ds = ds.shuffle(min(len(img_paths), 10000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(make_loader(task, enc, size), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)


# --------------------------------------------------------------------------- #
# Splitting and slice selection
# --------------------------------------------------------------------------- #
def split_indices(pairs, seed=SEED, val_frac=0.1, test_frac=0.1):
    rng = np.random.RandomState(seed)
    pids = [patient_id(ip, k) for ip, _, k in pairs]
    groups = {}
    for i, pid in enumerate(pids):
        groups.setdefault(pid, []).append(i)
    usable = None not in groups and 10 <= len(groups) <= len(pairs) // 5
    if usable:
        ids = sorted(groups)
        rng.shuffle(ids)
        n_val, n_test = max(1, int(len(ids) * val_frac)), max(1, int(len(ids) * test_frac))
        test_ids, val_ids = ids[:n_test], ids[n_test:n_test + n_val]
        tr, va, te = [], [], []
        for pid, idx in groups.items():
            (te if pid in test_ids else va if pid in val_ids else tr).extend(idx)
        print(f"[split] patient-level split over {len(ids)} scans -> "
              f"train {len(tr):,} | val {len(va):,} | test {len(te):,} slices")
    else:
        idx = np.arange(len(pairs))
        rng.shuffle(idx)
        n_test, n_val = int(len(idx) * test_frac), int(len(idx) * val_frac)
        te, va, tr = idx[:n_test].tolist(), idx[n_test:n_test + n_val].tolist(), idx[n_test + n_val:].tolist()
        print("[split] WARNING: could not infer patient IDs from file names, so this is a random "
              "slice-level split. Neighbouring slices from one patient can land in both train and "
              "val, which inflates validation scores.")
        print(f"[split] train {len(tr):,} | val {len(va):,} | test {len(te):,} slices")
    return np.array(tr), np.array(va), np.array(te)


def select_slices(idx, liver_px, tumor_px, task, seed=SEED):
    """Choose which slices a stage trains on, so empty slices don't dominate."""
    rng = np.random.RandomState(seed)
    idx = np.asarray(idx)
    if task == "liver":
        pos = idx[liver_px[idx] > 0]
        neg = idx[liver_px[idx] == 0]
    else:
        pos = idx[tumor_px[idx] > 0]
        neg = idx[(liver_px[idx] > 0) & (tumor_px[idx] == 0)]
    k = min(len(neg), int(len(pos) * (0.25 if task == "liver" else 1.0)))
    neg = rng.choice(neg, k, replace=False) if k else neg[:0]
    out = np.concatenate([pos, neg])
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def train_stage(task, args, pairs, enc, split, liver_px, tumor_px, out_dir):
    tr, va, _ = split
    tr = select_slices(tr, liver_px, tumor_px, task)
    va = select_slices(va, liver_px, tumor_px, task)
    if len(tr) == 0 or len(va) == 0:
        sys.exit(f"[{task}] no usable training/validation slices were found. "
                 f"Check the mask encoding printed above.")
    print(f"\n[{task}] training slices: {len(tr):,} | validation slices: {len(va):,}")

    def build(idx, training):
        return make_dataset([pairs[i][0] for i in idx], [pairs[i][1] for i in idx],
                            task, enc, args.img_size, args.batch_size, training)

    train_ds, val_ds = build(tr, True), build(va, False)
    model = build_unet((args.img_size, args.img_size, 1), args.base_filters, name=f"{task}_unet")
    epochs = args.epochs_liver if task == "liver" else args.epochs_tumor
    model.compile(optimizer=keras.optimizers.Adam(args.lr),
                  loss=liver_loss if task == "liver" else tumor_loss,
                  metrics=[dice_coef, iou_coef])
    path = out_dir / f"{task}_unet.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(path), monitor="val_dice_coef", mode="max",
                                        save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_dice_coef", mode="max", factor=0.5,
                                          patience=3, min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_dice_coef", mode="max", patience=8,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.CSVLogger(str(out_dir / f"{task}_history.csv")),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2)
    model.save(str(path))
    print(f"[{task}] saved -> {path}")
    return model


def cascade(liver_model, tumor_model, x, thr=0.5):
    """x: (B,H,W,1) in [0,1]. Returns boolean liver mask and tumor mask."""
    liver = liver_model(x, training=False).numpy() > thr
    tumor = tumor_model(x * liver.astype(np.float32), training=False).numpy() > thr
    return liver, tumor & liver


def evaluate_cascade(liver_model, tumor_model, pairs, enc, test_idx, args, out_dir):
    rng = np.random.RandomState(SEED)
    idx = np.array(test_idx)
    if len(idx) > args.eval_slices:
        idx = rng.choice(idx, args.eval_slices, replace=False)
    ds = make_dataset([pairs[i][0] for i in idx], [pairs[i][1] for i in idx],
                      "raw", enc, args.img_size, args.batch_size, False)
    acc = {"liver": [0.0, 0.0, 0.0], "tumor": [0.0, 0.0, 0.0]}  # inter, pred, gt
    for x, lbl in ds:
        lp, tp = cascade(liver_model, tumor_model, x.numpy())
        lbl = lbl.numpy()
        for name, pred, gt in (("liver", lp, lbl >= 1), ("tumor", tp, lbl == 2)):
            acc[name][0] += float((pred & gt).sum())
            acc[name][1] += float(pred.sum())
            acc[name][2] += float(gt.sum())
    res = {k: 2 * v[0] / (v[1] + v[2] + 1e-8) for k, v in acc.items()}
    print(f"\n[test] cascade Dice on {len(idx):,} held-out slices | "
          f"liver: {res['liver']:.4f} | tumor: {res['tumor']:.4f}")
    (out_dir / "test_metrics.json").write_text(json.dumps(
        {"test_slices": int(len(idx)), "dice_liver": res["liver"], "dice_tumor": res["tumor"]}, indent=2))
    return res


def save_sample_figure(liver_model, tumor_model, pairs, enc, test_idx, tumor_px, args, out_dir, n=6):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    test_idx = np.array(test_idx)
    cand = test_idx[tumor_px[test_idx] > 0]
    if len(cand) == 0:
        cand = test_idx
    cand = np.random.RandomState(SEED).choice(cand, min(n, len(cand)), replace=False)
    ds = make_dataset([pairs[i][0] for i in cand], [pairs[i][1] for i in cand],
                      "raw", enc, args.img_size, len(cand), False)
    x, lbl = next(iter(ds))
    lp, tp = cascade(liver_model, tumor_model, x.numpy())
    fig, ax = plt.subplots(len(cand), 3, figsize=(9, 3 * len(cand)), squeeze=False)
    for r in range(len(cand)):
        pred = lp[r, ..., 0].astype(np.uint8) + tp[r, ..., 0].astype(np.uint8)
        for c, (img, title) in enumerate(((x[r, ..., 0], "CT slice"),
                                          (lbl[r, ..., 0], "Ground truth"),
                                          (pred, "Prediction"))):
            ax[r, c].imshow(img, cmap="gray" if c == 0 else "viridis", vmin=0, vmax=1 if c == 0 else 2)
            ax[r, c].set_title(title, fontsize=9)
            ax[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "sample_predictions.png", dpi=120)
    plt.close(fig)
    print(f"[test] sample figure -> {out_dir / 'sample_predictions.png'}")


def run_predict(args, out_dir):
    liver_model = keras.models.load_model(out_dir / "liver_unet.keras",
                                          custom_objects=CUSTOM_OBJECTS, compile=False)
    tumor_model = keras.models.load_model(out_dir / "tumor_unet.keras",
                                          custom_objects=CUSTOM_OBJECTS, compile=False)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for f in args.predict:
        img = Image.open(f).convert("L").resize((args.img_size, args.img_size), Image.BILINEAR)
        x = np.asarray(img, dtype=np.float32)[None, ..., None] / 255.0
        lp, tp = cascade(liver_model, tumor_model, x)
        rgb = np.repeat(np.asarray(img)[..., None], 3, axis=-1).astype(np.float32)
        rgb[lp[0, ..., 0]] = 0.6 * rgb[lp[0, ..., 0]] + 0.4 * np.array([0, 255, 0])
        rgb[tp[0, ..., 0]] = 0.4 * rgb[tp[0, ..., 0]] + 0.6 * np.array([255, 0, 0])
        out = pred_dir / f"{Path(f).stem}_overlay.png"
        Image.fromarray(rgb.astype(np.uint8)).save(out)
        print(f"{f}: liver px={int(lp.sum())}, tumor px={int(tp.sum())} -> {out}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", help="Dataset root (default: auto-detect on Kaggle / ./dataset / ./data)")
    ap.add_argument("--images-dir", help="Image folder (use together with --masks-dir)")
    ap.add_argument("--masks-dir", help="Mask folder (use together with --images-dir)")
    ap.add_argument("--out-dir", help="Where to save models (default: /kaggle/working/models or ./models)")
    ap.add_argument("--only", choices=["liver", "tumor", "both"], default="both")
    ap.add_argument("--epochs-liver", type=int, default=30)
    ap.add_argument("--epochs-tumor", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--base-filters", type=int, default=32)
    ap.add_argument("--max-slices", type=int, default=0, help="Use a random subset (smoke test)")
    ap.add_argument("--eval-slices", type=int, default=4000, help="Max test slices for cascade evaluation")
    ap.add_argument("--mixed-precision", action="store_true", help="float16 compute on GPU (faster on T4/A100)")
    ap.add_argument("--predict", nargs="+", help="Run both saved models on these image files and exit")
    args, _ = ap.parse_known_args()
    return args


def main():
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print(f"[env] TensorFlow {tf.__version__} | GPUs: {len(tf.config.list_physical_devices('GPU'))}")
    if args.mixed_precision and tf.config.list_physical_devices("GPU"):
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("[env] mixed precision enabled")

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path("/kaggle/working/models") if Path("/kaggle/working").exists() else Path("models"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.predict:
        run_predict(args, out_dir)
        return

    pairs = discover_pairs(args)
    if args.max_slices and len(pairs) > args.max_slices:
        rng = random.Random(SEED)
        pairs = rng.sample(pairs, args.max_slices)
        print(f"[data] using a random subset of {len(pairs):,} slices (--max-slices)")

    enc = detect_encoding([m for _, m, _ in pairs])
    liver_px, tumor_px = scan_masks([m for _, m, _ in pairs], enc)
    split = split_indices(pairs)

    (out_dir / "dataset_report.json").write_text(json.dumps({
        "pairs": len(pairs), "mask_encoding": enc,
        "slices_with_liver": int((liver_px > 0).sum()),
        "slices_with_tumor": int((tumor_px > 0).sum()),
        "train": int(len(split[0])), "val": int(len(split[1])), "test": int(len(split[2])),
        "img_size": args.img_size,
    }, indent=2))

    liver_model = tumor_model = None
    if args.only in ("liver", "both"):
        liver_model = train_stage("liver", args, pairs, enc, split, liver_px, tumor_px, out_dir)
    if args.only in ("tumor", "both"):
        tumor_model = train_stage("tumor", args, pairs, enc, split, liver_px, tumor_px, out_dir)

    if liver_model is None and (out_dir / "liver_unet.keras").exists():
        liver_model = keras.models.load_model(out_dir / "liver_unet.keras",
                                              custom_objects=CUSTOM_OBJECTS, compile=False)
    if tumor_model is None and (out_dir / "tumor_unet.keras").exists():
        tumor_model = keras.models.load_model(out_dir / "tumor_unet.keras",
                                              custom_objects=CUSTOM_OBJECTS, compile=False)
    if liver_model is not None and tumor_model is not None and len(split[2]):
        evaluate_cascade(liver_model, tumor_model, pairs, enc, split[2], args, out_dir)
        save_sample_figure(liver_model, tumor_model, pairs, enc, split[2], tumor_px, args, out_dir)

    print(f"\nDone. Models in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
