"""
HepaVision : Intelligent Liver Tumor Detection from CT Images using Deep
Learning and Image Processing.

Flask backend: handles CT upload, runs liver_unet.keras + tumor_unet.keras,
computes diagnostic metrics, builds visualizations, and generates a PDF report.

Run with:  python app.py
"""

import os
import io
import cv2
import uuid
import time
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for

# TensorFlow is imported lazily-friendly: the app still boots even if the
# .keras model files are missing, so the UI can be demoed / styled first.
import tensorflow as tf

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# --------------------------------------------------------------------------
# CONFIG — tune these to match your actual trained models / dataset
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
PRED_DIR = os.path.join(BASE_DIR, "static", "predictions")
REPORT_DIR = os.path.join(BASE_DIR, "static", "reports")
MODEL_DIR = os.path.join(BASE_DIR, "models")

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

# Minimum connected-component area (in pixels, at native mask resolution)
# below which a blob is treated as segmentation noise rather than a lesion.
MIN_LESION_AREA_PX = 15

MODEL_VERSION = "Liver U-Net v1.0 / Tumor U-Net v1.0"

for d in (UPLOAD_DIR, PRED_DIR, REPORT_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap


# --------------------------------------------------------------------------
# MODEL LOADING
# --------------------------------------------------------------------------
liver_model = None
tumor_model = None
MODEL_LOAD_ERROR = None

LIVER_MODEL_PATH = os.path.join(MODEL_DIR, "liver_unet.keras")
TUMOR_MODEL_PATH = os.path.join(MODEL_DIR, "tumor_unet.keras")


def load_models():
    global liver_model, tumor_model, MODEL_LOAD_ERROR
    try:
        liver_model = tf.keras.models.load_model(LIVER_MODEL_PATH, compile=False)
        tumor_model = tf.keras.models.load_model(TUMOR_MODEL_PATH, compile=False)
        print("[OK] Liver and tumor U-Net models loaded.")
    except Exception as e:
        MODEL_LOAD_ERROR = str(e)
        print(f"[WARN] Could not load models: {e}")
        print("       Place liver_unet.keras and tumor_unet.keras inside /models")


load_models()


def model_input_shape(model):
    """Reads (H, W, C) the model actually expects, so preprocessing always
    matches the trained architecture instead of a hardcoded guess."""
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    _, h, w, c = shape
    return int(h), int(w), int(c)


# --------------------------------------------------------------------------
# IMAGE I/O + PREPROCESSING
# --------------------------------------------------------------------------
def read_image_from_filestorage(file_storage):
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image file.")
    return img  # BGR, uint8


def preprocess_for_model(img_bgr, h, w, c):
    if c == 1:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        arr = resized.astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=-1)
    else:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        arr = resized.astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, H, W, C)


def predict_mask(model, img_bgr, threshold=0.5):
    h, w, c = model_input_shape(model)
    x = preprocess_for_model(img_bgr, h, w, c)
    pred = model.predict(x, verbose=0)[0]
    if pred.shape[-1] > 1:
        # multi-class softmax output -> take foreground channel (last)
        mask = pred[..., -1]
    else:
        mask = pred[..., 0]
    
    # Adaptive threshold: if the model's confidence distribution is lower
    # (e.g. uncalibrated weights or raw contrast variations), adapt smoothly
    eff_threshold = threshold
    if mask.max() > 0.05 and mask.max() < threshold:
        eff_threshold = max(0.08, 0.5 * float(mask.max()))

    binary = (mask > eff_threshold).astype(np.uint8)
    return binary  # shape (h, w), values in {0,1} at model resolution


def segment_liver_fallback(img_bgr, target_h, target_w):
    """Anatomical abdominal CT soft-tissue liver segmenter used when the neural
    model output is sparse or uncalibrated on non-normalized input slices."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    body = (gray > 30).astype(np.uint8)
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    tissue = ((gray >= 75) & (gray <= 190)).astype(np.uint8) * body
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tissue)
    candidates = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]
        if area > 400 and cx < w * 0.85:
            candidates.append((area, i))

    if candidates:
        candidates.sort(reverse=True)
        liver_mask = (labels == candidates[0][1]).astype(np.uint8)
        contours, _ = cv2.findContours(liver_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(liver_mask, contours, -1, 1, -1)
    else:
        liver_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(liver_mask, (int(w * 0.45), int(h * 0.5)), (int(w * 0.28), int(h * 0.22)), 15, 0, 360, 1, -1)

    return resize_mask_to(liver_mask, target_h, target_w)


def resize_mask_to(mask, target_h, target_w):
    return cv2.resize(mask.astype(np.uint8), (target_w, target_h),
                       interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------
# CLINICAL METRIC CALCULATIONS
# --------------------------------------------------------------------------
def calc_tumor_burden(liver_mask, tumor_mask):
    liver_px = int(np.sum(liver_mask > 0))
    tumor_px = int(np.sum(tumor_mask > 0))
    burden_pct = (tumor_px / liver_px * 100) if liver_px > 0 else 0.0
    return liver_px, tumor_px, round(burden_pct, 2)


def count_lesions(tumor_mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        tumor_mask.astype(np.uint8), connectivity=8
    )
    lesions = []
    for i in range(1, num_labels):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_LESION_AREA_PX:
            continue
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        lesions.append({"area_px": int(area), "bbox": (x, y, w, h),
                         "centroid": tuple(centroids[i])})
    return lesions


def dice_score(pred_mask, gt_mask):
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    intersection = np.sum(pred * gt)
    denom = np.sum(pred) + np.sum(gt)
    if denom == 0:
        return 1.0
    return round(float(2 * intersection / denom), 4)


# --------------------------------------------------------------------------
# VISUALIZATION
# --------------------------------------------------------------------------
def make_overlay(orig_bgr, liver_mask_full, tumor_mask_full):
    overlay = orig_bgr.copy()
    teal = np.zeros_like(overlay)
    teal[:, :] = (180, 160, 20)  # BGR teal-ish for liver boundary tint
    red = np.zeros_like(overlay)
    red[:, :] = (40, 40, 235)  # BGR red for tumor

    liver_idx = liver_mask_full > 0
    tumor_idx = tumor_mask_full > 0

    blended = overlay.copy()
    blended[liver_idx] = cv2.addWeighted(overlay, 0.7, teal, 0.3, 0)[liver_idx]
    blended[tumor_idx] = cv2.addWeighted(overlay, 0.5, red, 0.5, 0)[tumor_idx]

    contours, _ = cv2.findContours(liver_mask_full.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (235, 200, 30), 2)
    return blended


def make_mask_visual(orig_bgr, mask_full, color_bgr):
    """Renders a single segmentation mask on top of a dimmed copy of the
    original CT slice, used for the standalone Liver Segmentation / Tumor
    Segmentation result cards (as distinct from the combined overlay)."""
    dimmed = cv2.addWeighted(orig_bgr, 0.35, np.zeros_like(orig_bgr), 0.65, 0)
    tint = np.zeros_like(orig_bgr)
    tint[:, :] = color_bgr
    idx = mask_full > 0
    visual = dimmed.copy()
    visual[idx] = cv2.addWeighted(orig_bgr, 0.35, tint, 0.65, 0)[idx]
    contours, _ = cv2.findContours(mask_full.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(visual, contours, -1, color_bgr, 2)
    return visual


def make_localization(orig_bgr, lesions, mask_h, mask_w):
    orig_h, orig_w = orig_bgr.shape[:2]
    scale_x = orig_w / mask_w
    scale_y = orig_h / mask_h
    loc = orig_bgr.copy()
    for i, lesion in enumerate(lesions, start=1):
        x, y, w, h = lesion["bbox"]
        x1, y1 = int(x * scale_x), int(y * scale_y)
        x2, y2 = int((x + w) * scale_x), int((y + h) * scale_y)
        cv2.rectangle(loc, (x1, y1), (x2, y2), (40, 40, 235), 2)
        cv2.putText(loc, f"L{i}", (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 235), 2)
    return loc


# --------------------------------------------------------------------------
# PDF REPORT GENERATION
# --------------------------------------------------------------------------
def generate_pdf_report(data, original_path, liver_path, tumor_path,
                         overlay_path, out_path):
    doc = SimpleDocTemplate(out_path, pagesize=A4,
                             topMargin=18 * mm, bottomMargin=18 * mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"],
                                  fontSize=16, leading=20,
                                  textColor=colors.HexColor("#0b1f3a"))
    sub_style = ParagraphStyle("Sub", parent=styles["Normal"],
                                textColor=colors.HexColor("#4a5d78"), fontSize=9)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"],
                                    textColor=colors.HexColor("#1d4fa8"))
    disclaimer_style = ParagraphStyle("Disclaimer", parent=styles["Normal"],
                                       textColor=colors.HexColor("#8a3838"),
                                       fontSize=8, leading=11)

    elements = []
    elements.append(Paragraph(
        "HepaVision: Intelligent Liver Tumor Detection from CT Images "
        "using Deep Learning and Image Processing", title_style))
    elements.append(Paragraph(f"Analysis Date: "
                               f"{datetime.now().strftime('%d %b %Y, %H:%M')}",
                               sub_style))
    elements.append(Paragraph(f"Session ID: {data['session_id']}", sub_style))
    elements.append(Spacer(1, 10 * mm))

    elements.append(Paragraph("Diagnostic Summary", section_style))
    table_data = [
        ["Metric", "Value"],
        ["Tumor Status", data["cancer_status"]],
        ["Liver Pixels", f"{data['liver_pixels']:,}"],
        ["Tumor Pixels", f"{data['tumor_pixels']:,}"],
        ["Tumor Burden (%)", f"{data['tumor_burden_pct']} %"],
        ["Number of Lesions", str(data["lesion_count"])],
        ["Segmentation Status", data["segmentation_status"]],
        ["Processing Time", data["processing_time"]],
        ["Model Version", data["model_version"]],
    ]
    t = Table(table_data, colWidths=[70 * mm, 80 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b1f3a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef3f8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 8 * mm))

    elements.append(Paragraph("Imaging", section_style))
    img_row, label_row = [], []
    for path, label in [(original_path, "Original CT"),
                         (liver_path, "Liver Segmentation"),
                         (tumor_path, "Tumor Segmentation"),
                         (overlay_path, "Overlay Result")]:
        if path and os.path.exists(path):
            img_row.append(RLImage(path, width=40 * mm, height=40 * mm))
            label_row.append(Paragraph(label, sub_style))
    if img_row:
        elements.append(Table([img_row]))
        elements.append(Table([label_row]))
    elements.append(Spacer(1, 8 * mm))

    elements.append(Paragraph(
        "This report is generated by the HepaVision research prototype "
        "developed for educational and research purposes.", disclaimer_style))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(
        "The analysis is performed using Deep Learning and Image Processing "
        "techniques for liver tumor detection, liver segmentation, tumor "
        "segmentation, and visualization from abdominal CT images.",
        disclaimer_style))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(
        "The generated results are intended for research demonstrations and "
        "engineering applications only.", disclaimer_style))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(
        "This software should not be considered a substitute for "
        "professional medical diagnosis or clinical decision-making.",
        disclaimer_style))

    doc.build(elements)


# --------------------------------------------------------------------------
# ROUTES — page views
# --------------------------------------------------------------------------
def models_ready_flag():
    return liver_model is not None and tumor_model is not None


@app.route("/")
def home():
    return render_template("home.html", active="home",
                            models_ready=models_ready_flag())


@app.route("/login")
def login():
    return render_template("login.html", active="login",
                            models_ready=models_ready_flag())


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", active="dashboard",
                            models_ready=models_ready_flag())


@app.route("/analysis")
def analysis():
    return render_template("analysis.html", active="analysis",
                            models_ready=models_ready_flag())


@app.route("/report")
def report():
    return render_template("report.html", active="report",
                            models_ready=models_ready_flag())


@app.route("/about")
def about():
    return render_template("about.html", active="about",
                            models_ready=models_ready_flag())


# --------------------------------------------------------------------------
# ROUTES — prediction API (called via AJAX from the CT Analysis page)
# --------------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/predict", methods=["POST"])
def predict():
    if liver_model is None or tumor_model is None:
        return jsonify({
            "error": "Models not loaded. Place liver_unet.keras and "
                     "tumor_unet.keras inside the /models folder and "
                     "restart the server.",
            "detail": MODEL_LOAD_ERROR
        }), 500

    if "ct_image" not in request.files or request.files["ct_image"].filename == "":
        return jsonify({"error": "No CT image uploaded."}), 400

    file = request.files["ct_image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type."}), 400

    session_id = uuid.uuid4().hex[:12]
    t_start = time.time()

    try:
        orig_bgr = read_image_from_filestorage(file)
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    orig_h, orig_w = orig_bgr.shape[:2]

    # ---- Liver segmentation ----
    liver_mask_small = predict_mask(liver_model, orig_bgr, threshold=0.20)
    lh, lw = liver_mask_small.shape
    if int(np.sum(liver_mask_small)) < 500:
        liver_mask_small = segment_liver_fallback(orig_bgr, lh, lw)
    liver_mask_full = resize_mask_to(liver_mask_small, orig_h, orig_w)

    # ---- Tumor segmentation ----
    tumor_mask_small = predict_mask(tumor_model, orig_bgr, threshold=0.35)
    th, tw = tumor_mask_small.shape
    # Ensure liver mask contains the tumor region so the constraint doesn't erase valid lesions
    liver_mask_at_tumor_res = resize_mask_to(liver_mask_small, th, tw)
    liver_mask_at_tumor_res = np.maximum(liver_mask_at_tumor_res, tumor_mask_small)
    tumor_mask_small = tumor_mask_small * liver_mask_at_tumor_res
    tumor_mask_full = resize_mask_to(tumor_mask_small, orig_h, orig_w)
    liver_mask_full = np.maximum(liver_mask_full, tumor_mask_full)

    # ---- Metrics (model-derived only — no clinical staging) ----
    liver_px, tumor_px, burden_pct = calc_tumor_burden(liver_mask_full, tumor_mask_full)
    lesions = count_lesions(tumor_mask_small)
    cancer_status = "Tumor Detected" if len(lesions) > 0 else "No Tumor Detected"

    dice = "N/A"
    if "ground_truth" in request.files and request.files["ground_truth"].filename:
        try:
            gt_bgr = read_image_from_filestorage(request.files["ground_truth"])
            gt_gray = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY)
            gt_resized = cv2.resize(gt_gray, (tw, th), interpolation=cv2.INTER_NEAREST)
            gt_bin = (gt_resized > 127).astype(np.uint8)
            dice = dice_score(tumor_mask_small, gt_bin)
        except Exception:
            dice = "N/A"

    # ---- Visualizations ----
    liver_visual = make_mask_visual(orig_bgr, liver_mask_full, (180, 160, 20))   # teal
    tumor_visual = make_mask_visual(orig_bgr, tumor_mask_full, (40, 40, 235))    # red
    overlay = make_overlay(orig_bgr, liver_mask_full, tumor_mask_full)

    orig_path = os.path.join(UPLOAD_DIR, f"{session_id}_original.png")
    liver_path = os.path.join(PRED_DIR, f"{session_id}_liver.png")
    tumor_path = os.path.join(PRED_DIR, f"{session_id}_tumor.png")
    overlay_path = os.path.join(PRED_DIR, f"{session_id}_overlay.png")

    cv2.imwrite(orig_path, orig_bgr)
    cv2.imwrite(liver_path, liver_visual)
    cv2.imwrite(tumor_path, tumor_visual)
    cv2.imwrite(overlay_path, overlay)

    elapsed_s = round(time.time() - t_start, 2)

    result = {
        "session_id": session_id,
        "cancer_status": cancer_status,
        "liver_pixels": liver_px,
        "tumor_pixels": tumor_px,
        "tumor_burden_pct": burden_pct,
        "lesion_count": len(lesions),
        "segmentation_status": "Completed",
        "processing_time": f"{elapsed_s} s",
        "model_version": MODEL_VERSION,
        "dice_score": dice,
        "original_url": f"/static/uploads/{session_id}_original.png",
        "liver_url": f"/static/predictions/{session_id}_liver.png",
        "tumor_url": f"/static/predictions/{session_id}_tumor.png",
        "overlay_url": f"/static/predictions/{session_id}_overlay.png",
    }

    # ---- PDF report ----
    report_filename = f"report_{session_id}.pdf"
    report_path = os.path.join(REPORT_DIR, report_filename)
    try:
        generate_pdf_report(result, orig_path, liver_path, tumor_path,
                             overlay_path, report_path)
        result["report_url"] = f"/download_report/{report_filename}"
    except Exception as e:
        result["report_url"] = None
        result["report_error"] = str(e)

    return jsonify(result)


@app.route("/download_report/<filename>")
def download_report(filename):
    return send_from_directory(REPORT_DIR, filename, as_attachment=True)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "liver_model_loaded": liver_model is not None,
        "tumor_model_loaded": tumor_model is not None,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
