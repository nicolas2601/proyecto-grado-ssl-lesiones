(() => {
  const state = { report: null, file: null, stream: null, toastTimer: 0, live: false, liveBusy: false, lastRequestStarted: 0 };
  const LIVE_MIN_INTERVAL_MS = 3000;
  const REQUEST_TIMEOUT_MS = 30000;
  const $ = (selector) => document.querySelector(selector);
  const configuredApiOrigin = document.querySelector('meta[name="api-origin"]')?.content?.trim();
  const apiOrigin = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? window.location.origin
    : configuredApiOrigin || window.location.origin;
  const apiUrl = (path) => new URL(path, `${apiOrigin}/`).toString();
  const classNames = {
    akiec: 'Queratosis actínica / carcinoma intraepitelial',
    bcc: 'Carcinoma basocelular',
    bkl: 'Lesión benigna tipo queratosis',
    df: 'Dermatofibroma',
    mel: 'Melanoma',
    nv: 'Nevus melanocítico',
    vasc: 'Lesión vascular',
  };
  const preview = $('#preview-image');
  const video = $('#video');
  const input = $('#file-input');
  const analyzeButton = $('#analyze-button');
  const liveButton = $('#live-button');
  const toast = $('#toast');

  function showMessage(message) {
    toast.textContent = message;
    toast.dataset.visible = 'true';
    window.clearTimeout(state.toastTimer);
    state.toastTimer = window.setTimeout(() => { toast.dataset.visible = 'false'; }, 4800);
  }

  function getSelectedModel() {
    return document.querySelector('input[name="model"]:checked')?.value || '';
  }

  function modelIsAvailable(model = getSelectedModel()) {
    return Boolean(model && state.report?.models?.[model]?.classification_ready && navigator.onLine !== false);
  }

  function updateAnalyzeButton() {
    const available = modelIsAvailable();
    analyzeButton.disabled = !available || !state.file || state.live;
    liveButton.disabled = !available || !state.stream;
    $('#disabled-note').textContent = !navigator.onLine
      ? 'Conéctate al servidor; la app no realiza inferencia sin conexión.'
      : available && state.file
        ? 'Listo para enviar una imagen al servidor.'
        : 'Selecciona un modelo disponible y una imagen para continuar.';
  }

  function setFile(file) {
    if (!file) return;
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      showMessage('Formato no compatible. Elige una imagen JPEG, PNG o WebP.');
      input.value = '';
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showMessage('La imagen supera el límite de 10 MB.');
      input.value = '';
      return;
    }
    stopCamera();
    state.file = file;
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    video.hidden = true;
    $('#empty-state').hidden = true;
    $('#preview-label').hidden = false;
    $('#result').hidden = true;
    $('#image-hint').textContent = `${file.name || 'Captura de cámara'} · ${(file.size / 1048576).toFixed(2)} MB. Se envía al servidor; no se guarda en esta app.`;
    updateAnalyzeButton();
  }

  async function openCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      showMessage('La cámara requiere HTTPS o localhost y permiso del navegador.');
      return;
    }
    try {
      stopCamera();
      state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' } }, audio: false });
      video.srcObject = state.stream;
      video.hidden = false;
      preview.hidden = true;
      await video.play();
      $('#camera-actions').hidden = false;
      $('#camera-button').hidden = true;
      updateAnalyzeButton();
    } catch (error) {
      state.stream = null;
      showMessage(error.name === 'NotAllowedError' ? 'Permite el acceso a la cámara en la configuración del navegador.' : 'No se pudo abrir la cámara. También puedes subir una fotografía.');
    }
  }

  function stopCamera() {
    stopLiveCapture();
    if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
    video.srcObject = null;
    video.hidden = true;
    $('#camera-actions').hidden = true;
    $('#camera-button').hidden = false;
    if (!state.file) preview.hidden = false;
    updateAnalyzeButton();
  }

  function captureFrame() {
    if (!video.videoWidth || !video.videoHeight) {
      showMessage('Espera a que la vista previa de cámara esté lista.');
      return;
    }
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (!blob) {
        showMessage('No fue posible capturar la imagen.');
        return;
      }
      setFile(new File([blob], `captura-${Date.now()}.jpg`, { type: 'image/jpeg' }));
    }, 'image/jpeg', 0.92);
  }

  function renderPrediction(data, model, live = false) {
    if (!Array.isArray(data.classes) || data.classes.length !== 7 || !data.probs || typeof data.pred !== 'string'
      || new Set(data.classes).size !== 7 || data.classes.some((code) => !Object.hasOwn(classNames, code))
      || !data.classes.includes(data.pred)) {
      throw new Error('Respuesta de clasificación incompleta.');
    }
    const scores = data.classes.map((code) => ({ code, score: Number(data.probs[code]) }));
    if (scores.some(({ score }) => !Number.isFinite(score) || score < 0 || score > 1)) {
      throw new Error('El servidor devolvió puntuaciones inválidas.');
    }
    scores.sort((left, right) => right.score - left.score);
    const topThree = $('#top-three');
    topThree.replaceChildren();
    scores.slice(0, 3).forEach(({ code, score }) => {
      const item = document.createElement('li');
      item.dataset.code = code;
      item.textContent = `${classNames[code] || code} · ${(score * 100).toFixed(1).replace('.', ',')} %`;
      topThree.append(item);
    });
    $('#melanoma-score').textContent = `${(Number(data.probs.mel) * 100).toFixed(1).replace('.', ',')} %`;
    $('#result-model').textContent = `${model} · ${data.mode || 'base'}`;
    $('#prediction-name').textContent = classNames[data.pred] || data.pred;
    const leadingScore = Number(data.probs[data.pred]);
    $('#prediction-score').textContent = `${(leadingScore * 100).toFixed(1).replace('.', ',')} %`;
    const list = $('#probability-list');
    list.replaceChildren();
    scores.forEach(({ code, score }) => {
      const row = document.createElement('div');
      row.className = 'probability-row';
      const label = document.createElement('span');
      label.className = 'probability-label';
      label.textContent = classNames[code] || code;
      const track = document.createElement('span');
      track.className = 'probability-track';
      track.setAttribute('aria-hidden', 'true');
      const fill = document.createElement('span');
      fill.className = 'probability-fill';
      fill.dataset.leading = String(code === data.pred);
      fill.style.transform = `scaleX(${score})`;
      track.append(fill);
      const value = document.createElement('span');
      value.className = 'probability-value';
      value.textContent = `${(score * 100).toFixed(1).replace('.', ',')} %`;
      row.append(label, track, value);
      list.append(row);
    });
    const identity = data.identity || {};
    $('#model-identity').textContent = [identity.model_id || model, identity.release, identity.backbone_sha256 ? `SHA-256 ${identity.backbone_sha256}` : null].filter(Boolean).join(' · ');
    const cycle = $('#cycle-note');
    cycle.textContent = data.cycle != null
      ? `Ciclo de inferencia reportado por el servidor: ${data.cycle}.`
      : 'Resultado de una solicitud puntual al servidor; no es análisis continuo de video.';
    cycle.hidden = false;
    $('#result').hidden = false;
    if (!live) $('#result').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'nearest' });
  }

  function apiErrorMessage(status, code, fallback) {
    if (code === 'RATE_LIMITED' || status === 429) return 'Se alcanzó la cuota compartida de la demo. Captura en vivo pausada; espera antes de volver a intentarlo.';
    if (status === 413) return 'Imagen demasiado grande. El servidor admite hasta 10 MB y 20 megapíxeles; elige una imagen más pequeña.';
    if (status === 408 || status === 504) return 'La carga o el procesamiento superó el tiempo límite. Verifica la conexión y vuelve a intentar.';
    return fallback || 'No se pudo procesar la imagen.';
  }

  async function postPrediction(file, model) {
    const body = new FormData();
    body.append('file', file, file.name || 'captura.jpg');
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    state.lastRequestStarted = Date.now();
    try {
      const response = await fetch(apiUrl(`/api/predict?model=${encodeURIComponent(model)}&mode=base`), {
        method: 'POST', body, cache: 'no-store', signal: controller.signal,
      });
      const data = await response.json();
      if (!response.ok) {
        const error = new Error(apiErrorMessage(response.status, data?.error?.code, data?.error?.message));
        error.status = response.status;
        error.code = data?.error?.code;
        throw error;
      }
      return data;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('La solicitud superó los 30 segundos. Revisa la conexión; no se enviará otra automáticamente.');
      throw error;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  async function analyze(file = state.file, { live = false } = {}) {
    const model = getSelectedModel();
    if (!file || !model || !modelIsAvailable(model)) return null;
    if (state.liveBusy) return null;
    state.liveBusy = true;
    if (!live) {
      analyzeButton.disabled = true;
      analyzeButton.setAttribute('aria-busy', 'true');
      analyzeButton.firstElementChild.textContent = 'Procesando imagen…';
    }
    try {
      const data = await postPrediction(file, model);
      if (getSelectedModel() !== model) return null;
      renderPrediction(data, model, live);
      return data;
    } catch (error) {
      showMessage(error.message || 'No se pudo conectar con el servidor. Revisa tu conexión e intenta de nuevo.');
      if (error.status === 429 || error.code === 'RATE_LIMITED') stopLiveCapture({ quota: true });
      return { error };
    } finally {
      state.liveBusy = false;
      if (!live) {
        analyzeButton.removeAttribute('aria-busy');
        analyzeButton.firstElementChild.textContent = 'Analizar imagen';
      }
      updateAnalyzeButton();
    }
  }

  function startLiveCapture() {
    if (!state.stream || !modelIsAvailable() || state.live || state.liveBusy) return;
    state.live = true;
    liveButton.setAttribute('aria-pressed', 'true');
    $('#live-label').textContent = 'Pausar captura en vivo';
    updateAnalyzeButton();
    void runLiveCapture();
  }

  function stopLiveCapture({ quota = false } = {}) {
    state.live = false;
    liveButton?.setAttribute('aria-pressed', 'false');
    const label = $('#live-label');
    if (label) label.textContent = 'Iniciar captura en vivo';
    if (liveButton) liveButton.dataset.paused = 'true';
    if (quota) showMessage('Se alcanzó la cuota compartida de la demo. Captura en vivo pausada; espera antes de volver a intentarlo.');
  }

  async function runLiveCapture() {
    while (state.live && state.stream && navigator.onLine !== false) {
      if (!video.videoWidth || !video.videoHeight) {
        showMessage('Vista previa no lista; la captura en vivo se pausó.');
        stopLiveCapture();
        break;
      }
      const elapsed = Date.now() - state.lastRequestStarted;
      if (state.lastRequestStarted && elapsed < LIVE_MIN_INTERVAL_MS) {
        await new Promise((resolve) => window.setTimeout(resolve, LIVE_MIN_INTERVAL_MS - elapsed));
        if (!state.live) break;
      }
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.82));
      if (!blob || !state.live) {
        if (!blob) showMessage('No fue posible capturar el fotograma; captura en vivo pausada.');
        stopLiveCapture();
        break;
      }
      const result = await analyze(new File([blob], `ciclo-${Date.now()}.jpg`, { type: 'image/jpeg' }), { live: true });
      if (!result || result.error) {
        stopLiveCapture();
        break;
      }
      liveButton.dataset.paused = 'false';
      await new Promise((resolve) => window.setTimeout(resolve, Math.max(0, LIVE_MIN_INTERVAL_MS - (Date.now() - state.lastRequestStarted))));
    }
    updateAnalyzeButton();
  }

  async function loadReport() {
    const connection = $('#connection');
    try {
      const response = await fetch(apiUrl('/api/report'), { cache: 'no-store' });
      if (!response.ok) throw new Error('report unavailable');
      state.report = await response.json();
      ['A71', 'B71'].forEach((key) => {
        const model = state.report.models?.[key];
        const ready = model?.classification_ready === true;
        const radio = document.querySelector(`input[name="model"][value="${key}"]`);
        radio.disabled = !ready;
        const status = $(`#status-${key}`);
        status.textContent = ready ? 'Disponible' : 'No disponible';
        status.dataset.ready = String(ready);
        const identity = model?.identity || {};
        const identityNote = $(`#${key}-identity`);
        identityNote.replaceChildren();
        if (identity.backbone_sha256) {
          const text = document.createElement('p');
          const digest = identity.backbone_sha256;
          const shortDigest = digest.length > 24 ? `${digest.slice(0, 12)}…${digest.slice(-8)}` : digest;
          text.textContent = `${key} · SHA-256 ${shortDigest}${identity.sha256_verified === false ? ' (declarado, no verificado)' : ''}`;
          identityNote.append(text);
          identityNote.hidden = false;
        } else {
          identityNote.hidden = true;
        }
      });
      connection.dataset.state = 'ready';
      const anyReady = ['A71', 'B71'].some((key) => state.report.models?.[key]?.classification_ready === true);
      connection.textContent = anyReady ? 'Servidor conectado' : 'Modelos no disponibles';
      updateAnalyzeButton();
    } catch {
      connection.dataset.state = 'offline';
      connection.textContent = 'Sin conexión';
      ['A71', 'B71'].forEach((key) => {
        document.querySelector(`input[name="model"][value="${key}"]`).disabled = true;
        $(`#status-${key}`).textContent = 'No disponible';
      });
      stopLiveCapture();
      $('#disabled-note').textContent = 'Conéctate al servidor; la app no realiza inferencia sin conexión.';
      updateAnalyzeButton();
    }
  }

  input.addEventListener('change', () => setFile(input.files?.[0]));
  $('#upload-button').addEventListener('click', () => input.click());
  document.querySelectorAll('input[name="model"]').forEach((radio) => radio.addEventListener('change', () => {
    stopLiveCapture();
    $('#result').hidden = true;
    updateAnalyzeButton();
  }));
  $('#camera-button').addEventListener('click', openCamera);
  $('#capture-button').addEventListener('click', captureFrame);
  $('#cancel-camera').addEventListener('click', stopCamera);
  analyzeButton.addEventListener('click', () => analyze());
  liveButton.addEventListener('click', () => state.live ? stopLiveCapture() : startLiveCapture());
  window.addEventListener('pagehide', stopCamera);
  window.addEventListener('online', loadReport);
  window.addEventListener('offline', () => {
    $('#connection').dataset.state = 'offline';
    $('#connection').textContent = 'Sin conexión';
    stopLiveCapture();
    updateAnalyzeButton();
  });
  if ('serviceWorker' in navigator) window.addEventListener('load', () => navigator.serviceWorker.register('/sw.js', { scope: '/' }).catch(() => {}));
  loadReport();
})();
