// Evaluation preprocessing of the v7 classifier in plain JavaScript.
//
// Reproduces PIL Image.resize(BILINEAR) bit for bit (Pillow's two-pass
// fixed-point resampler, 22 precision bits, uint8 between passes), then
// torchvision CenterCrop(224) and ImageNet normalisation in float32. Given the
// same decoded RGB pixels it returns what preprocess.py returns; the browser's
// JPEG decoder may still differ from libjpeg by a few intensity levels.

export const RESIZE = 255;
export const CROP = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const PRECISION_BITS = 22;

export function resizedSize(width, height, size = RESIZE) {
  if (width <= height) return [size, Math.trunc((size * height) / width)];
  return [Math.trunc((size * width) / height), size];
}

function bilinear(x) {
  const ax = Math.abs(x);
  return ax < 1 ? 1 - ax : 0;
}

function coefficients(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Int32Array(outSize * ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize) - xmin;
    const weights = new Float64Array(xmax);
    let total = 0;
    for (let x = 0; x < xmax; x++) {
      weights[x] = bilinear((x + xmin - center + 0.5) / filterscale);
      total += weights[x];
    }
    for (let x = 0; x < xmax; x++) {
      const w = total !== 0 ? weights[x] / total : weights[x];
      kk[xx * ksize + x] = Math.trunc(w < 0 ? -0.5 + w * (1 << PRECISION_BITS) : 0.5 + w * (1 << PRECISION_BITS));
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { bounds, kk, ksize };
}

function clip8(value) {
  if (value >= 1 << (PRECISION_BITS + 8)) return 255;
  if (value <= 0) return 0;
  return value >> PRECISION_BITS;
}

// One separable pass over RGB uint8 data; `horizontal` picks the axis.
function resamplePass(src, width, height, outSize, horizontal) {
  const inSize = horizontal ? width : height;
  const { bounds, kk, ksize } = coefficients(inSize, outSize);
  const outW = horizontal ? outSize : width;
  const outH = horizontal ? height : outSize;
  const out = new Uint8Array(outW * outH * 3);
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      const i = horizontal ? x : y;
      const xmin = bounds[i * 2];
      const n = bounds[i * 2 + 1];
      for (let c = 0; c < 3; c++) {
        let ss = 1 << (PRECISION_BITS - 1);
        for (let k = 0; k < n; k++) {
          const sx = horizontal ? xmin + k : x;
          const sy = horizontal ? y : xmin + k;
          ss += src[(sy * width + sx) * 3 + c] * kk[i * ksize + k];
        }
        out[(y * outW + x) * 3 + c] = clip8(ss);
      }
    }
  }
  return out;
}

// PIL Image.resize((outW, outH), BILINEAR) on packed RGB bytes.
export function resizeBilinear(rgb, width, height, outW, outH) {
  let data = rgb;
  let w = width;
  if (outW !== width) {
    data = resamplePass(data, w, height, outW, true);
    w = outW;
  }
  if (outH !== height) data = resamplePass(data, w, height, outH, false);
  return data;
}

// Python round(): halves go to the even neighbour (torchvision CenterCrop uses it).
function roundHalfEven(x) {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff !== 0.5) return Math.round(x);
  return floor % 2 === 0 ? floor : floor + 1;
}

// Packed RGB bytes (width x height) -> Float32Array (3, 224, 224), NCHW order.
export function preprocessRGB(rgb, width, height) {
  const [rw, rh] = resizedSize(width, height);
  const resized = resizeBilinear(rgb, width, height, rw, rh);
  const top = roundHalfEven((rh - CROP) / 2);
  const left = roundHalfEven((rw - CROP) / 2);
  const out = new Float32Array(3 * CROP * CROP);
  for (let c = 0; c < 3; c++) {
    const mean = Math.fround(MEAN[c]);
    const std = Math.fround(STD[c]);
    for (let y = 0; y < CROP; y++) {
      for (let x = 0; x < CROP; x++) {
        const v = resized[((y + top) * rw + (x + left)) * 3 + c];
        const scaled = Math.fround(v / 255);
        out[c * CROP * CROP + y * CROP + x] = Math.fround(Math.fround(scaled - mean) / std);
      }
    }
  }
  return out;
}

// RGBA bytes from a canvas (alpha ignored, as PIL convert("RGB") does for opaque images).
export function rgbaToRgb(rgba) {
  const rgb = new Uint8Array((rgba.length / 4) * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    rgb[j] = rgba[i];
    rgb[j + 1] = rgba[i + 1];
    rgb[j + 2] = rgba[i + 2];
  }
  return rgb;
}
