const state = { report: null, stream: null, selectedBlob: null, liveLoop: null, liveBusy: false };

const $ = (selector) => document.querySelector(selector);

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatMetric(value) {
  return Number(value).toFixed(3).replace(".", ",");
}

function formatInterval(interval) {
  return `[${formatMetric(interval[0])}, ${formatMetric(interval[1])}]`;
}

function formatInteger(value) {
  return new Intl.NumberFormat("es-CO").format(Number(value));
}

const backboneNames = {
  dinov2_public_vits14: "DINOv2 público S/14",
  dinov2_public_vitb14: "DINOv2 público B/14",
  ssl_plain_vits14: "Plain S/14",
  ssl_plain_vitb14: "Plain B/14",
  ssl_proposed_vits14: "Propuesto S/14",
  ssl_proposed_vitb14: "Propuesto B/14",
};

const classNames = {
  akiec: { name: "Queratosis actínica / carcinoma intraepitelial", code: "akiec" },
  bcc: { name: "Carcinoma basocelular", code: "bcc" },
  bkl: { name: "Lesión benigna tipo queratosis", code: "bkl" },
  df: { name: "Dermatofibroma", code: "df" },
  mel: { name: "Melanoma", code: "mel" },
  nv: { name: "Nevus melanocítico (lunar común)", code: "nv" },
  vasc: { name: "Lesión vascular", code: "vasc" },
};

function displayArm(name) {
  return backboneNames[name] || name;
}

function renderModelStatus(model) {
  const ready = model.classification_ready;
  const backboneReady = model.backbone_ready;
  $("#status-title").textContent = ready ? "Clasificador listo" : backboneReady ? "Backbone listo" : "Checkpoint pendiente";
  $("#status-copy").textContent = ready
    ? "El backbone y el probe estan disponibles para probar una imagen."
    : backboneReady
      ? "El .pt carga como extractor. Falta el probe de clasificacion publicado."
      : "Configura el .pt y su run.json para activar la inferencia local.";
  $("#status-led").classList.toggle("ready", ready || backboneReady);
  $("#release-tag").textContent = state.report.source.release.toUpperCase();
  $("#probe-backbone").textContent = backboneReady ? "LISTO" : "PENDIENTE";
  $("#probe-classifier").textContent = ready ? "LISTO" : "PENDIENTE";
  $("#probe-mode").textContent = ready ? "CONTINUO" : "EXTRACTOR";
  $("#probe-backbone").classList.toggle("ready", backboneReady);
  $("#probe-classifier").classList.toggle("ready", ready);
  const liveButton = $("#live-button");
  liveButton.disabled = !ready;
  liveButton.textContent = ready ? "Seguir en vivo" : "En vivo: requiere clasificador";
  $("#hero-disclaimer").textContent = state.report.disclaimer;
}

function renderBars(classification) {
  const rows = [
    ["DINOv2 publico B", classification.dinov2_public_vitb14.macro_f1, ""],
    ["Plain B / control", classification.ssl_plain_vitb14.macro_f1, ""],
    ["Proposed B / tono", classification.ssl_proposed_vitb14.macro_f1, "proposed"],
  ];
  $("#classification-bars").innerHTML = rows.map(([label, value, kind]) => `
    <div class="bar-row ${kind}">
      <span class="bar-label">${escapeHtml(label)}</span>
      <progress class="bar-meter" max="0.7" value="${Number(value)}">${formatMetric(value)}</progress>
      <span class="bar-value">${formatMetric(value)}</span>
    </div>`).join("");
}

function renderDatasetRail(report) {
  const datasets = report.datasets;
  $("#dataset-rail").innerHTML = [
    [datasets.pool_total, "imagenes en el pool sin etiquetas"],
    [datasets.pool_valid, `entraron al tono (${datasets.pool_valid_pct} %)`],
    [datasets.quarantined, "en cuarentena por mascara"],
  ].map(([value, label]) => `<div class="dataset-chip"><strong>${formatInteger(value)}</strong><span>${escapeHtml(label)}</span></div>`).join("");
}

function renderSegmentation(report) {
  const segmentation = report.segmentation;
  $("#segmentation-note").textContent = segmentation.note;
  $("#segmentation-grid").innerHTML = [segmentation.isic2018, segmentation.ham10000].map((item) => `
    <article class="segmentation-card ${item.role === "externo" ? "external" : ""}">
      <span>${escapeHtml(item.role)} · ${formatInteger(item.n)} imagenes</span>
      <strong>${formatMetric(item.dice)}</strong>
      <small>Dice medio · IC95 % ${formatInterval(item.ci95)} · HD95 mediana ${formatMetric(item.hd95_median_px)} px</small>
    </article>`).join("");
  const tone = segmentation.tone_equity;
  $("#segmentation-tone").innerHTML = `
    <div><h3>Equidad de segmentacion</h3><p>El tercil claro–oscuro cae de forma sólida en HAM10000. En ISIC 2018 el contraste no queda confirmado.</p></div>
    <div class="tone-gap"><span>HAM10000 · gap Dice</span><strong>${formatMetric(tone.ham10000.gap_dice)}</strong></div>
    <div class="tone-gap"><span>ISIC 2018 · gap Dice</span><strong>${formatMetric(tone.isic2018.gap_dice)}</strong></div>`;
}

function renderTraining(report) {
  $("#training-note").textContent = report.training.note;
  $("#training-table-body").innerHTML = report.training.arms.map((arm) => `
    <tr><td>${escapeHtml(displayArm(arm.arm))}</td><td>${escapeHtml(arm.adaptation_mode)}</td><td>${escapeHtml(arm.initial_weights)}</td><td>${formatMetric(arm.budget_gpu_hours)}</td><td>${formatInteger(arm.steps)}</td><td>${formatInteger(arm.samples)}</td></tr>`).join("");
}

function renderClassificationTable(report) {
  $("#classification-table-body").innerHTML = report.classification_table.map((row) => {
    const proposed = row.backbone.startsWith("ssl_proposed");
    return `<tr class="${proposed ? "proposed-row" : ""}">
      <td>${escapeHtml(displayArm(row.backbone))}</td>
      <td>${formatMetric(row.macro_f1)}</td><td>${formatMetric(row.balanced_accuracy)}</td>
      <td>${formatMetric(row.auc)}</td><td>${formatMetric(row.ece)}</td>
      <td>${proposed ? "brazo propuesto" : row.backbone.includes("public") ? "referencia" : "control"}</td>
    </tr>`;
  }).join("");
}

function contrastLabel(row) {
  const arm = displayArm(row.arm);
  const control = displayArm(row.control);
  return `${arm} − ${control}`;
}

function renderContrasts(report) {
  $("#contrast-note").textContent = `${report.contrast_meta.n_resamples} remuestras · pareado por clase · seed ${report.contrast_meta.rng_seed}`;
  $("#contrast-table-body").innerHTML = report.contrasts.map((row) => {
    const established = row.excludes_zero;
    const verdict = !established
      ? "No se distingue del ruido"
      : row.direction === "arm_better" ? `Gana ${displayArm(row.arm)}` : `Gana ${displayArm(row.control)}`;
    const verdictClass = !established ? "neutral" : row.direction === "arm_better" ? "winner" : "loss";
    return `<tr><td>${escapeHtml(contrastLabel(row))}</td><td>${escapeHtml(row.metric)}</td><td>${formatMetric(row.delta)}</td><td>[${formatMetric(row.ci95_low)}, ${formatMetric(row.ci95_high)}]</td><td class="${verdictClass}">${escapeHtml(verdict)}</td></tr>`;
  }).join("");
}

function renderEquity(report) {
  $("#equity-copy").innerHTML = `<h3>No se afirma efecto.</h3><p>Los intervalos de los estratos extremos se solapan en los seis backbones. La brecha descriptiva cambia, pero no alcanza para decir que el tono la causa o la corrige.</p>`;
  $("#equity-table-body").innerHTML = Object.entries(report.classification_tone_equity).map(([name, data]) => `
    <tr><td>${escapeHtml(displayArm(name))}</td><td>${formatMetric(data.gap_macro_f1)}</td><td>${escapeHtml(data.worst_label)}</td><td class="neutral">${data.gap_established ? "sí" : "no"}</td></tr>`).join("");
}

function renderDownloads(report) {
  const release = report.site_meta.release;
  $("#download-rail").innerHTML = `<div><strong>Release ${escapeHtml(report.downloads.release_tag)}</strong><span>Backbones, run.json, probe y checksums forman el paquete reproducible.</span></div><a href="${escapeHtml(release)}" target="_blank" rel="noreferrer">Abrir release ↗</a>`;
  $("#caveat-list").innerHTML = `<p class="caveat-title">Antes de interpretar una salida</p>${report.caveats.map((caveat) => `<p>• ${escapeHtml(caveat)}</p>`).join("")}`;
}

function renderReport(report) {
  const proposed = report.classification.ssl_proposed_vitb14;
  const tone = report.tone;
  const segmentation = report.segmentation;
  renderBars(report.classification);
  renderDatasetRail(report);
  renderSegmentation(report);
  renderTraining(report);
  renderClassificationTable(report);
  renderContrasts(report);
  renderEquity(report);
  renderDownloads(report);
  $("#proposed-auc").textContent = formatMetric(proposed.auc);
  $("#auc-delta").textContent = `Δ ${formatMetric(report.classification_verdict.proposed_vs_public_auc.delta)} vs publico`;
  $("#proposed-ece").textContent = formatMetric(proposed.ece);
  $("#tone-gap").textContent = formatMetric(tone.classification_gap_proposed_vitb14);
  $("#isic-dice").textContent = formatMetric(segmentation.isic2018.dice);
  $("#isic-ci").textContent = `IC95 % ${formatInterval(segmentation.isic2018.ci95)}`;
  $("#ham-dice").textContent = formatMetric(segmentation.ham10000.dice);
  $("#ham-ci").textContent = `IC95 % ${formatInterval(segmentation.ham10000.ci95)}`;
  $("#roadmap-list").innerHTML = report.next_steps.map((step, index) => `
    <article class="roadmap-item"><span>${String(index + 1).padStart(2, "0")}</span><div><h3>${escapeHtml(step.title)}</h3><p>${escapeHtml(step.detail)}</p></div></article>`).join("");
  renderModelStatus(report.model);
}

function setResult(title, copy, detail = "") {
  $("#result-state").textContent = title;
  $("#result-copy").textContent = copy;
  $("#result-detail").textContent = detail;
  $("#benchmark-result").hidden = true;
}

function showPreview(blob) {
  const image = $("#image-preview");
  image.src = URL.createObjectURL(blob);
  image.hidden = false;
  $("#camera-stage").classList.remove("camera-live");
  $("#stage-placeholder").hidden = true;
  $("#capture-button").disabled = false;
}

async function uploadForInference(blob, filename = "camera.jpg", { live = false } = {}) {
  if (!blob) return;
  if (live && state.liveBusy) return;
  if (live) state.liveBusy = true;
  $("#camera-stage").classList.add("scanning");
  if (!live) setResult("Procesando", "La imagen se transforma en una representacion local.");
  const endpoint = state.report.model.classification_ready ? "/api/predict?mode=tta" : "/api/features";
  const form = new FormData();
  form.append("file", blob, filename);
  try {
    const response = await fetch(endpoint, { method: "POST", body: form });
    const body = await response.json();
    if (!response.ok) {
      const error = body.error || {};
      setResult(live ? "Continuo detenido" : "No listo", error.message || "El servicio no pudo procesar la imagen.", error.code || "ERROR");
      if (live) stopLiveLoop();
      return;
    }
    if (endpoint === "/api/features") {
      if (!live) setResult("Features extraidas", "El backbone responde, pero esta entrega aun no trae cabeza de clasificacion.", `dimension: ${body.feature_dim} · norma: ${body.feature_norm}`);
      return;
    }
    renderPrediction(body);
  } catch (error) {
    setResult("Sin conexion", "No se pudo contactar al backend local.", error.message);
  } finally {
    $("#camera-stage").classList.remove("scanning");
    if (live) state.liveBusy = false;
  }
}

function renderPrediction(prediction) {
  const predictedClass = classNames[prediction.pred] || { name: prediction.pred, code: prediction.pred };
  setResult(
    predictedClass.name,
    `Clase más probable: ${predictedClass.name}. Confianza estimada ${formatMetric(prediction.confidence)}.`,
    `código: ${predictedClass.code} · ${prediction.inference_mode === "tta_3views" ? "3 vistas" : "vista base"} · vector: ${prediction.feature_dim} dimensiones`,
  );
  if (prediction.benchmark) {
    const trueClass = classNames[prediction.benchmark.true_class] || {
      name: prediction.benchmark.true_class,
      code: prediction.benchmark.true_class,
    };
    const isCorrect = prediction.benchmark.true_class === prediction.pred;
    const benchmarkResult = $("#benchmark-result");
    benchmarkResult.hidden = false;
    benchmarkResult.classList.toggle("match", isCorrect);
    benchmarkResult.innerHTML = `<span>CLASE REAL DEL GOLDEN</span><strong>${escapeHtml(trueClass.name)}</strong><small>${escapeHtml(trueClass.code)} · ${isCorrect ? "top-1 correcto" : "el modelo eligió otra clase"}</small>`;
  }
  $("#prob-list").innerHTML = Object.entries(prediction.probs).map(([label, value]) => `
    <div class="prob-row" title="${escapeHtml(classNames[label]?.name || label)}">
      <span class="prob-label"><strong>${escapeHtml(classNames[label]?.name || label)}</strong><small>${escapeHtml(classNames[label]?.code || label)}</small></span>
      <div class="prob-track"><progress class="prob-progress" max="1" value="${Number(value)}">${formatMetric(value)}</progress></div>
      <span class="prob-value">${formatMetric(value)}</span>
    </div>`).join("");
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setResult("Camara no disponible", "Este navegador no expone getUserMedia en el contexto actual.");
    return;
  }
  try {
    state.stream?.getTracks().forEach((track) => track.stop());
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false });
    const video = $("#camera-video");
    video.srcObject = state.stream;
    $("#camera-stage").classList.add("camera-live");
    $("#image-preview").hidden = true;
    $("#stage-placeholder").hidden = true;
    $("#capture-button").disabled = false;
    $("#camera-button").textContent = "Camara activa";
    $("#capture-mode").textContent = "CAMARA EN VIVO";
    setResult("Camara lista", "Captura un frame para enviarlo al backend local.");
  } catch (error) {
    setResult("Permiso pendiente", "El navegador rechazo o no concedio acceso a la camara.", error.message);
  }
}

function stopLiveLoop() {
  if (state.liveLoop) window.clearInterval(state.liveLoop);
  state.liveLoop = null;
  const liveButton = $("#live-button");
  liveButton.setAttribute("aria-pressed", "false");
  liveButton.textContent = state.report?.model.classification_ready ? "Seguir en vivo" : "En vivo: requiere clasificador";
  $("#capture-mode").textContent = "CAMARA EN VIVO";
}

function captureFrame({ live = false } = {}) {
  const video = $("#camera-video");
  if (!$("#camera-stage").classList.contains("camera-live")) {
    uploadForInference(state.selectedBlob, "selected-image.jpg", { live });
    return;
  }
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob((blob) => {
    if (blob) {
      state.selectedBlob = blob;
      if (!live) showPreview(blob);
      uploadForInference(blob, "camera-frame.jpg", { live });
    }
  }, "image/jpeg", .92);
}

function toggleLiveLoop() {
  if (state.liveLoop) {
    stopLiveLoop();
    return;
  }
  if (!state.report?.model.classification_ready || !$("#camera-stage").classList.contains("camera-live")) {
    setResult("Probe requerido", "Activa la camara y configura el probe para clasificar en continuo.");
    return;
  }
  const liveButton = $("#live-button");
  liveButton.setAttribute("aria-pressed", "true");
  liveButton.textContent = "Detener análisis en vivo";
  $("#capture-mode").textContent = "CLASIFICACION CONTINUA";
  setResult("En vivo", "Analizando un frame cada 2,2 segundos. La camara permanece en el navegador.");
  captureFrame({ live: true });
  state.liveLoop = window.setInterval(() => captureFrame({ live: true }), 2200);
}

function setupReveals() {
  const targets = document.querySelectorAll(".section-intro, .section-heading-row, .verdict-card, .metric-panel, .stat-tile, .segmentation-strip, .pipeline-list li, .lab-grid, .roadmap-item, .closing-section > *");
  targets.forEach((element, index) => {
    element.classList.add("reveal");
    element.dataset.revealOrder = String((index % 5) + 1);
  });
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    targets.forEach((element) => element.classList.add("is-visible"));
    return;
  }
  const observer = new IntersectionObserver((entries, currentObserver) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("is-visible");
      currentObserver.unobserve(entry.target);
    });
  }, { threshold: .12, rootMargin: "0px 0px -8%" });
  targets.forEach((element) => observer.observe(element));
}

async function boot() {
  try {
    const response = await fetch("/api/report");
    state.report = await response.json();
    renderReport(state.report);
  } catch (error) {
    setResult("Reporte no disponible", "El backend no pudo cargar los artefactos versionados.", error.message);
  }
  $("#camera-button").addEventListener("click", startCamera);
  $("#capture-button").addEventListener("click", captureFrame);
  $("#live-button").addEventListener("click", toggleLiveLoop);
  $("#image-input").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    state.selectedBlob = file;
    showPreview(file);
    $("#capture-mode").textContent = "ARCHIVO SELECCIONADO";
    uploadForInference(file, file.name);
  });
  setupReveals();
  window.addEventListener("beforeunload", () => {
    stopLiveLoop();
    state.stream?.getTracks().forEach((track) => track.stop());
  });
}

boot();
