// =============================================================================
// HepaVision — frontend interactivity
// =============================================================================

document.addEventListener("DOMContentLoaded", () => {
  initClock();
  initUpload();
});

// ---------------------------------------------------------------------------
// Live date (used on dashboard / analysis topbars)
// ---------------------------------------------------------------------------
function initClock() {
  const el = document.getElementById("liveDate");
  if (!el) return;
  const now = new Date();
  el.textContent = now.toLocaleDateString(undefined, {
    weekday: "short", year: "numeric", month: "short", day: "numeric"
  });
}

// ---------------------------------------------------------------------------
// Upload + drag/drop + preview + analysis run (CT Analysis page only)
// ---------------------------------------------------------------------------
let selectedFile = null;

function initUpload() {
  const dropZone = document.getElementById("dropZone");
  if (!dropZone) return; // not on this page

  const fileInput = document.getElementById("ctImageInput");
  const previewRow = document.getElementById("previewRow");
  const previewImg = document.getElementById("previewImg");
  const previewName = document.getElementById("previewName");
  const clearBtn = document.getElementById("clearBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const form = document.getElementById("uploadForm");
  const newAnalysisBtn = document.getElementById("newAnalysisBtn");

  dropZone.addEventListener("click", () => fileInput.click());

  ["dragenter", "dragover"].forEach(evt =>
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach(evt =>
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    })
  );
  dropZone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      previewRow.style.display = "flex";
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  clearBtn.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    previewRow.style.display = "none";
    analyzeBtn.disabled = true;
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!selectedFile) return;
    await runAnalysis();
  });

  if (newAnalysisBtn) {
    newAnalysisBtn.addEventListener("click", () => {
      document.getElementById("resultsContent").style.display = "none";
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }
}

// ---------------------------------------------------------------------------
// Analysis run: pipeline animation + AJAX call to /predict + render results
// ---------------------------------------------------------------------------
async function runAnalysis() {
  const errorBox = document.getElementById("errorBox");
  errorBox.style.display = "none";

  const track = document.getElementById("pipelineTrack");
  track.classList.add("show");
  animatePipeline();

  const formData = new FormData();
  formData.append("ct_image", selectedFile);
  const gtInput = document.getElementById("gtInput");
  if (gtInput.files[0]) formData.append("ground_truth", gtInput.files[0]);

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Analysis failed.");
    }

    finishPipeline();
    data.timestamp = new Date().toLocaleString();
    renderResults(data);
    saveToHistory(data);

    document.getElementById("resultsContent").style.display = "block";
    setTimeout(() => {
      document.getElementById("resultsContent").scrollIntoView({ behavior: "smooth", block: "start" });
    }, 300);
  } catch (err) {
    track.classList.remove("show");
    errorBox.textContent = err.message;
    errorBox.style.display = "block";
  }
}

function animatePipeline() {
  const steps = document.querySelectorAll(".pt-step");
  steps.forEach(s => s.classList.remove("active", "done"));
  let i = 0;
  steps[0].classList.add("active");
  window._pipelineInterval = setInterval(() => {
    steps[i].classList.remove("active");
    steps[i].classList.add("done");
    i = Math.min(i + 1, steps.length - 1);
    if (i < steps.length) steps[i].classList.add("active");
  }, 550);
}

function finishPipeline() {
  clearInterval(window._pipelineInterval);
  document.querySelectorAll(".pt-step").forEach(s => {
    s.classList.remove("active");
    s.classList.add("done");
  });
}

// ---------------------------------------------------------------------------
// Render results into the CT Analysis page
// ---------------------------------------------------------------------------
function renderResults(data) {
  document.getElementById("resultTimestamp").textContent = data.timestamp;

  const badge = document.getElementById("cancerStatusBadge");
  const positive = data.cancer_status === "Tumor Detected";
  badge.textContent = data.cancer_status;
  badge.className = "status-badge " + (positive ? "positive" : "negative");

  document.getElementById("imgOriginal").src = data.original_url;
  document.getElementById("imgLiver").src = data.liver_url || data.overlay_url;
  document.getElementById("imgTumor").src = data.tumor_url || data.localization_url;
  document.getElementById("imgOverlay").src = data.overlay_url;

  animateCount(document.getElementById("metricBurden"), data.tumor_burden_pct, "%");
  document.getElementById("gaugeBurden").style.width = Math.min(data.tumor_burden_pct, 100) + "%";

  animateCount(document.getElementById("metricLesions"), data.lesion_count, "");

  document.getElementById("metricPixels").textContent =
    `${data.liver_pixels.toLocaleString()} / ${data.tumor_pixels.toLocaleString()}`;

  document.getElementById("metricSegStatus").textContent = data.segmentation_status || "Completed";
  document.getElementById("metricProcTime").textContent = data.processing_time || "—";
  document.getElementById("metricModelVersion").textContent = data.model_version || "—";
  document.getElementById("metricModelStatus").textContent = "Loaded Successfully";
  document.getElementById("metricDice").textContent = data.dice_score;

  const reportBtn = document.getElementById("downloadReportBtn");
  if (data.report_url) {
    reportBtn.href = data.report_url;
    reportBtn.style.opacity = 1;
    reportBtn.style.pointerEvents = "auto";
  } else {
    reportBtn.style.opacity = 0.5;
    reportBtn.style.pointerEvents = "none";
  }
}

// ---------------------------------------------------------------------------
// Persist the last result (for the Report page) + a lightweight history
// (for the Dashboard "Recent Analyses" list). Client-side only.
// ---------------------------------------------------------------------------
function saveToHistory(data) {
  localStorage.setItem("hepavision_last_result", JSON.stringify(data));

  const history = JSON.parse(localStorage.getItem("hepavision_history") || "[]");
  history.unshift({
    session_id: data.session_id,
    cancer_status: data.cancer_status,
    tumor_burden_pct: data.tumor_burden_pct,
    timestamp: data.timestamp
  });
  localStorage.setItem("hepavision_history", JSON.stringify(history.slice(0, 10)));
}

// Simple count-up animation for numeric readouts
function animateCount(el, target, suffix) {
  const duration = 700;
  const start = performance.now();
  const isFloat = target % 1 !== 0;

  function frame(now) {
    const progress = Math.min((now - start) / duration, 1);
    const value = target * progress;
    el.textContent = (isFloat ? value.toFixed(2) : Math.round(value)) + suffix;
    if (progress < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
