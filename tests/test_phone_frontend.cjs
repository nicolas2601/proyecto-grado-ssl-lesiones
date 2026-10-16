const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const root = path.join(__dirname, '../demo/phone');

function loadApp() {
  const elements = new Map();
  const radios = ['A71', 'B71'].map((value) => ({ value, disabled: false, addEventListener() {} }));
  const document = {
    querySelector(selector) {
      if (selector === 'input[name="model"]:checked') return radios.find((radio) => radio.checked) || null;
      const match = selector.match(/input\[name="model"\]\[value="(A71|B71)"\]/);
      if (match) return radios.find((radio) => radio.value === match[1]);
      if (!elements.has(selector)) elements.set(selector, makeElement());
      return elements.get(selector);
    },
    querySelectorAll(selector) { return selector === 'input[name="model"]' ? radios : []; },
    createElement(tag) { return makeElement(tag); },
  };
  const context = vm.createContext({
    document,
    window: {
      location: { hostname: 'dermssl-phone.vercel.app', origin: 'https://dermssl-phone.vercel.app' },
      clearTimeout() {}, setTimeout() { return 1; }, addEventListener() {},
    },
    navigator: { mediaDevices: {}, onLine: true },
    URL: { createObjectURL: () => 'blob:test' },
    FormData: class { append() {} },
    File: class {},
    matchMedia: () => ({ matches: true }),
    fetch: async () => ({ ok: false, json: async () => ({}) }),
    console,
    setTimeout,
    clearTimeout,
  });
  return { context, elements, radios };
}

function makeElement() {
  return {
    hidden: false, disabled: false, dataset: {}, style: {}, children: [], textContent: '', value: '',
    classList: { add() {}, remove() {} },
    addEventListener() {}, setAttribute() {}, removeAttribute() {}, scrollIntoView() {},
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    get firstElementChild() { return this.children[0] || (this._child ||= { textContent: '' }); },
  };
}

test('mobile app requests only A71/B71 base classification and exposes seven scores', async () => {
  const source = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
  const { context, elements } = loadApp();
  const app = source.replace(/\n  loadReport\(\);\n\}\)\(\);\s*$/, '\n})();');
  vm.runInContext(app, context);
  context.document = undefined;
  const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
  assert.match(html, /id="analyze-button"[^>]*disabled/);
  assert.match(html, /value="A71"/);
  assert.match(html, /value="B71"/);
  assert.doesNotMatch(html, /value="legacy"|value="public"/);
  assert.match(source, /apiUrl\(`\/api\/predict\?model=\$\{encodeURIComponent\(model\)\}&mode=base`\)/);
  assert.match(source, /data\.classes\.length !== 7/);
  assert.match(source, /identity\.backbone_sha256/);
  assert.match(source, /meta\[name="api-origin"\]/);
  assert.match(source, /window\.location\.hostname === 'localhost'/);
  assert.ok(elements);
});

test('service worker only caches the explicit shell and bypasses API and POST requests', () => {
  const sw = fs.readFileSync(path.join(root, 'sw.js'), 'utf8');
  assert.match(sw, /request\.method !== 'GET'/);
  assert.match(sw, /url\.pathname\.startsWith\('\/api\/'\)/);
  assert.match(sw, /SHELL\.includes\(url\.pathname\)/);
  assert.match(sw, /self\.registration\.scope/);
  assert.doesNotMatch(sw, /\/api\/report|\/api\/predict/);
});

test('PWA shell links only to intended architecture context and names experimental limits', () => {
  const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
  assert.match(html, /phone-research\.html/);
  assert.match(html, /experiment-gates\.html/);
  assert.match(html, /split-firewall\.html/);
  assert.match(html, /Herramienta auxiliar/);
  assert.match(html, /Puntuaciones softmax del clasificador; no representan probabilidad clínica/);
  assert.match(html, /Segmentación: <strong>pendiente<\/strong>/);
  assert.match(html, /no describen B71/);
  assert.match(html, /pad_dev_H1_paired\.json/);
  assert.match(html, /702d086e202cb9045cc159534280efc7d9c70597406453fd6ca68ec9d066dd8e/);
  assert.match(html, /Segmentación F4: pendiente/);
  assert.match(html, /reporte histórico del laboratorio no representa métricas actuales/);
  assert.match(html, /derm\.nicolasmoreno\.site\/docs/);
  assert.match(html, /reporte de investigación \(histórico\)/i);
  assert.match(html, /name="api-origin" content="https:\/\/derm\.nicolasmoreno\.site"/);
  assert.match(html, /href="\/manifest\.webmanifest"/);
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'manifest.webmanifest'), 'utf8'));
  assert.equal(manifest.start_url, '/');
  assert.equal(manifest.scope, '/');
  assert.match(fs.readFileSync(path.join(root, 'app.js'), 'utf8'), /register\('\/sw\.js', \{ scope: '\/' \}\)/);
  for (const asset of ['/styles.css', '/app.js', '/icon.svg']) assert.ok(html.includes(`"${asset}"`));
  for (const size of ['192x192', '512x512']) assert.ok(manifest.icons.some((icon) => icon.sizes === size && icon.type === 'image/png'));
});

test('live capture cannot overlap requests and respects the minimum three-second start gap', () => {
  const source = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
  assert.match(source, /function startLiveCapture\(/);
  assert.match(source, /LIVE_MIN_INTERVAL_MS = 3000/);
  assert.match(source, /await analyze\(/);
  assert.match(source, /RATE_LIMITED/);
});

test('upload timeout and rate limit errors explain the shared API quota', () => {
  const source = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
  assert.match(source, /REQUEST_TIMEOUT_MS = 30000/);
  assert.match(source, /status === 408/);
  assert.match(source, /status === 413/);
  assert.match(source, /status === 429/);
  assert.match(source, /cuota compartida/);
});

test('upload is keyboard operable and probability fills have real geometry', () => {
  const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
  assert.match(html, /<button[^>]*id="upload-button"[^>]*type="button"/);
  const css = fs.readFileSync(path.join(root, 'styles.css'), 'utf8');
  assert.match(css, /\.probability-fill\s*\{[^}]*display:\s*block;[^}]*width:\s*100%/);
});

test('inference cannot overlap and a late result cannot cross model selection', async () => {
  const source = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
  const { context, radios } = loadApp();
  context.URL = URL;
  context.AbortController = AbortController;
  let calls = 0;
  let resolveRequest;
  context.fetch = () => { calls += 1; return new Promise((resolve) => { resolveRequest = resolve; }); };
  vm.runInContext(source.replace(/\n  loadReport\(\);\n\}\)\(\);\s*$/, '\n  globalThis.testApi = { state, analyze };\n})();'), context);
  radios[0].checked = true;
  context.testApi.state.report = { models: { A71: { classification_ready: true }, B71: { classification_ready: true } } };
  const first = context.testApi.analyze({ name: 'benchmark.jpg' });
  assert.equal(await context.testApi.analyze({ name: 'benchmark.jpg' }), null);
  assert.equal(calls, 1);
  radios[0].checked = false;
  radios[1].checked = true;
  resolveRequest({ ok: true, json: async () => ({}) });
  assert.equal(await first, null);
  assert.equal(context.testApi.state.liveBusy, false);
});

for (const melanomaScore of [0.01, 0.3]) {
  test(`results keep seven scores, top three and a separate melanoma score (${melanomaScore})`, () => {
    const source = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
    const { context, elements } = loadApp();
    vm.runInContext(source.replace(/\n  loadReport\(\);\n\}\)\(\);\s*$/, '\n  globalThis.renderPrediction = renderPrediction;\n})();'), context);
    const probs = { nv: 0.4, bcc: 0.2, bkl: 0.15, mel: melanomaScore, df: 0.02, akiec: 0.02, vasc: 0.21 - melanomaScore };
    if (melanomaScore === 0.3) { probs.bcc = 0.1; probs.bkl = 0.1; probs.vasc = 0.06; }
    context.renderPrediction({ classes: Object.keys(probs), probs, pred: 'nv' }, 'B71');
    assert.equal(elements.get('#probability-list').children.length, 7);
    const expected = Object.entries(probs).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([code]) => code);
    assert.deepEqual(elements.get('#top-three').children.map((row) => row.dataset.code), expected);
    assert.equal(elements.get('#melanoma-score').textContent, `${(melanomaScore * 100).toFixed(1).replace('.', ',')} %`);
    const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
    assert.match(html, /Puede confundir melanoma con nevus; no descarta melanoma/);
    assert.match(html, /puntuación del modelo, no riesgo clínico/i);
  });
}

test('results reject missing melanoma and unknown prediction instead of showing an invalid score', () => {
  const source = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
  const { context } = loadApp();
  vm.runInContext(source.replace(/\n  loadReport\(\);\n\}\)\(\);\s*$/, '\n  globalThis.renderPrediction = renderPrediction;\n})();'), context);
  const probs = { nv: 0.4, bcc: 0.2, bkl: 0.1, df: 0.1, akiec: 0.1, vasc: 0.05, unknown: 0.05 };
  assert.throws(() => context.renderPrediction({ classes: Object.keys(probs), probs, pred: 'nv' }, 'B71'), /incompleta/);
  probs.mel = probs.unknown;
  delete probs.unknown;
  assert.throws(() => context.renderPrediction({ classes: Object.keys(probs), probs, pred: 'unknown' }, 'B71'), /incompleta/);
});
