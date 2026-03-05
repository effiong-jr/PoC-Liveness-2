const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const sdkDir = path.join(root, 'node_modules', 'faceplugin-face-recognition-js');
const ortDir = path.join(root, 'node_modules', 'onnxruntime-web', 'dist');
const publicDir = path.join(root, 'public');

function copy(src, dest) {
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.copyFileSync(src, dest);
  console.log(`Copied: ${path.relative(root, dest)}`);
}

// Only the two models used by the liveness detection flow.
// (The other 7 models in the SDK package are not called by this app.)
const models = ['fr_detect.onnx', 'fr_liveness.onnx'];
for (const m of models) {
  copy(path.join(sdkDir, 'model', m), path.join(publicDir, 'model', m));
}

// OpenCV JS loader + WASM binary
const cvFiles = ['opencv.js', 'opencv_js.wasm'];
for (const f of cvFiles) {
  copy(path.join(sdkDir, 'js', f), path.join(publicDir, 'js', f));
}

// ORT WASM backend files.
// ORT 1.20+ uses two variants:
//  - jsep  : primary wasm execution provider (loaded first)
//  - plain : fallback if jsep fails
// asyncify and jspi are for unrelated use-cases (Asyncify/JSPI) and are never
// loaded by this app, so we skip them to reduce deployment size.
const ortFiles = [
  'ort-wasm-simd-threaded.jsep.mjs',
  'ort-wasm-simd-threaded.jsep.wasm',
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.wasm',
];
for (const f of ortFiles) {
  copy(path.join(ortDir, f), path.join(publicDir, f));
}

console.log('\nEssential model and WASM files copied to /public (~42 MB total).');
