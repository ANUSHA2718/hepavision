---
title: HepaVision
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 5000
---

# HepaVision
### Intelligent Liver Tumor Detection from CT Images using Deep Learning and Image Processing

A Flask + TensorFlow web application for a two-stage liver/tumor segmentation
pipeline (`liver_unet.keras` → `tumor_unet.keras`), with tumor burden
quantification, lesion counting, and automatic PDF report generation. The
frontend has been fully redesigned as a premium, multi-page glassmorphism
medical-AI experience while the Flask backend, TensorFlow integration,
prediction pipeline, and report generation are unchanged in behavior.

## Pages

| Route         | Page          | Description                                              |
|---------------|---------------|------------------------------------------------------------|
| `/`           | Home          | Landing page, project overview, technology badges           |
| `/login`      | Login         | UI-only login screen (no auth enforced) → redirects to Dashboard |
| `/dashboard`  | Dashboard     | Model status, quick actions, recent analyses (client-side)  |
| `/analysis`   | CT Analysis   | Upload CT scan, live pipeline animation, results & metrics  |
| `/report`     | Report        | Premium hospital-style diagnostic report + PDF download     |
| `/about`      | About         | Project overview, methodology, research objectives          |

There is no Contact page, contact form, or footer contact information.

## Folder structure

```
hepavision/
├── app.py
├── requirements.txt
├── templates/
│   ├── _public_base.html   (shared shell for Home / Login)
│   ├── _app_base.html      (shared sidebar shell for Dashboard / Analysis / Report / About)
│   ├── home.html
│   ├── login.html
│   ├── dashboard.html
│   ├── analysis.html
│   ├── report.html
│   └── about.html
├── static/
│   ├── style.css
│   ├── script.js
│   ├── uploads/        (uploaded CT images saved here)
│   ├── predictions/    (liver / tumor / overlay images saved here)
│   └── reports/        (generated PDF reports saved here)
└── models/
    ├── liver_unet.keras
    └── tumor_unet.keras
```

## Setup

1. Place your trained `liver_unet.keras` and `tumor_unet.keras` files inside
   the `models/` folder (exact filenames matter — `app.py` looks for these
   two names).
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   python app.py
   ```

4. Open `http://localhost:5000` in your browser.

If the model files aren't found, the site still loads (you can review the
full UI), but the sidebar shows "Models Offline" and `/predict` returns a
clear error until the `.keras` files are in place.

## API

`POST /predict` is unchanged in purpose: it accepts `ct_image` (required)
and an optional `ground_truth` mask, runs the liver → tumor U-Net pipeline,
and returns JSON with:

- `cancer_status`, `liver_pixels`, `tumor_pixels`, `tumor_burden_pct`,
  `lesion_count`, `segmentation_status`, `processing_time`, `model_version`,
  `dice_score`
- `original_url`, `liver_url`, `tumor_url`, `overlay_url` — the four result
  images shown on the CT Analysis page
- `report_url` — link to the generated PDF

All fields are produced directly by the trained models and image-processing
pipeline — nothing is hardcoded or fabricated. Clinical staging (TNM,
Stage I–IV, tumor size in cm) has been intentionally removed from both the
UI and the PDF report, since it is not a model output.

`GET /download_report/<filename>` and `GET /health` are unchanged.

## Frontend notes

- The CT Analysis and Report pages read/write results via a small amount of
  client-side JavaScript (`localStorage`) so the Dashboard's "Recent
  Analyses" list and the Report page can show your latest run without any
  backend session/database changes.
- All visuals (liver blueprint grid, radiology rings, floating pipeline
  cards) are generated with CSS/inline SVG — no external or copyrighted
  artwork is used.

## Optional: Dice score

The CT Analysis page has an optional "ground-truth mask" file input. If you
upload a binary tumor mask alongside the CT image, the app computes the Dice
coefficient between the predicted and ground-truth tumor masks. If omitted,
Dice score shows as `N/A`.

## Disclaimer

HepaVision is a research and educational prototype. It is not a certified
medical device and must not be used for real clinical diagnosis or treatment
decisions.
