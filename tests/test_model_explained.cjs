const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const root = path.join(__dirname, '../demo/phone');
const html = fs.readFileSync(path.join(root, 'model-explained.html'), 'utf8');

test('explainer is linked from the phone app and uses its shared design system', () => {
  const index = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
  assert.match(index, /href="\/model-explained\.html"[^>]*>\s*<span>Cómo piensa el modelo<\/span>/);
  assert.match(html, /<html lang="es">/);
  assert.match(html, /href="\/styles\.css"/);
  assert.match(html, /id="main"/);
  assert.match(html, /class="skip-link"/);
});

test('claims separate local historical evidence from pending mobile evaluation', () => {
  assert.match(html, /8\.011/);
  assert.match(html, /2\.004/);
  assert.match(html, /A71_classifier_manifest\.json/);
  assert.match(html, /B71_classifier_manifest\.json/);
  assert.match(html, /Macro-F1/);
  assert.match(html, /HAM val ya consultado/);
  assert.match(html, /balanced accuracy, matriz 7×7/);
  assert.match(html, /no medido/);
  assert.match(html, /balanced accuracy, matriz 7×7/);
});

test('model identities and safety limits avoid equating A71, B71, and D0', () => {
  assert.match(html, /A71 − D0/);
  assert.match(html, /D0<\/th><td>DINOv2 público/);
  assert.match(html, /f075194fd84ddaa4041ebd5225ae8e3cad643b41a3195d9b927e2ca27087b12b/);
  assert.match(html, /a392ebe8eedcc519a6131e3930bed5ea232d4be7084010f82740e55cadbe39c1/);
  assert.match(html, /misma receta documentada de sonda, pero cada uno tiene coeficientes/);
  assert.match(html, /568 imágenes y 353 pacientes/);
  assert.match(html, /27 imágenes de melanoma de 18 pacientes/);
  assert.match(html, /ficha global PAD contiene 1\.141 imágenes de 686 pacientes/);
  assert.match(html, /No existe sonda balanced promovida a producción/);
  assert.match(html, /No uses estos resultados para diagnosticar/);
  assert.match(html, /no los convierte en probabilidad de enfermedad/);
});

test('class glossary covers all seven registered labels and table headers are semantic', () => {
  for (const label of ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']) assert.match(html, new RegExp(`<code>${label}<\\/code>`));
  assert.match(html, /<caption>Clases registradas y conteos disponibles del conjunto<\/caption>/);
  assert.ok((html.match(/<th scope="col">/g) || []).length >= 3);
});

test('explainer styles the authored information hierarchy and evidence tables', () => {
  const css = fs.readFileSync(path.join(root, 'styles.css'), 'utf8');
  for (const selector of ['.explain-layout', '.pipeline', '.class-table', '.identity-table', '.evidence-table', '.glossary']) {
    assert.ok(css.includes(selector), `Missing scoped explainer styles for ${selector}`);
  }
});
