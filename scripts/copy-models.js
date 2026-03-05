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

// Copy ONNX model weights
const models = [
  'fr_detect.onnx',
  'fr_liveness.onnx',
  'fr_age.onnx',
  'fr_expression.onnx',
  'fr_eye.onnx',
  'fr_feature.onnx',
  'fr_gender.onnx',
  'fr_landmark.onnx',
  'fr_pose.onnx',
];
for (const m of models) {
  copy(path.join(sdkDir, 'model', m), path.join(publicDir, 'model', m));
}

// Copy OpenCV files
const cvFiles = ['opencv.js', 'opencv_js.wasm'];
for (const f of cvFiles) {
  copy(path.join(sdkDir, 'js', f), path.join(publicDir, 'js', f));
}

// Copy ONNX Runtime Web WASM binaries and MJS loader modules.
// ORT 1.20+ splits the WASM backend into .wasm binaries AND .mjs JS loaders;
// both must be served statically from the page origin so that
// `import('ort-wasm-simd-threaded.jsep.mjs')` succeeds in the browser.
const ortFiles = fs
  .readdirSync(ortDir)
  .filter((f) => f.endsWith('.wasm') || f.endsWith('.mjs'));
for (const f of ortFiles) {
  copy(path.join(ortDir, f), path.join(publicDir, f));
}

console.log('\nAll model, WASM, and ORT loader files copied to /public successfully.');
