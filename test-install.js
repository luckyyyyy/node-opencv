/**
 * npm install smoke test
 *
 * Run this after `npm install node-opencv-rs` to verify the package works:
 *   node test-install.js
 *
 * Or run against the local build:
 *   node test-install.js --local
 */
'use strict';

const path = require('path');
const assert = require('assert');

const isLocal = process.argv.includes('--local');
const cv = isLocal
  ? require('.')
  : require('node-opencv-rs');

async function main() {
  console.log(`Testing ${isLocal ? 'local build' : 'node-opencv-rs'} …`);

  // 1. Mat creation
  const m = cv.Mat.zeros(4, 4, cv.CV_8UC3);
  assert.strictEqual(m.rows, 4, 'rows');
  assert.strictEqual(m.cols, 4, 'cols');
  assert.strictEqual(m.channels, 3, 'channels');
  assert.ok(!m.empty, 'not empty');
  console.log('  ✓ Mat.zeros');

  // 2. fromBuffer + data round-trip
  const buf = Buffer.alloc(3, 42);
  const m1 = cv.Mat.fromBuffer(1, 1, cv.CV_8UC3, buf);
  assert.strictEqual(m1.data[0], 42, 'data[0]');
  assert.strictEqual(m1.data[1], 42, 'data[1]');
  assert.strictEqual(m1.data[2], 42, 'data[2]');
  console.log('  ✓ Mat.fromBuffer / data');

  // 3. imdecode / imencode round-trip (in-memory)
  const src = cv.Mat.zeros(8, 8, cv.CV_8UC3);
  const encoded = await cv.imencode('.png', src);
  assert.ok(encoded instanceof Buffer && encoded.length > 0, 'imencode returned Buffer');
  const decoded = await cv.imdecode(encoded);
  assert.strictEqual(decoded.rows, 8, 'decoded rows');
  assert.strictEqual(decoded.cols, 8, 'decoded cols');
  console.log('  ✓ imencode / imdecode');

  // 4. cvtColor BGR → GRAY
  const color = cv.Mat.zeros(4, 4, cv.CV_8UC3);
  const gray = await color.cvtColor(cv.ColorCode.Bgr2Gray);
  assert.strictEqual(gray.channels, 1, 'gray channels');
  console.log('  ✓ cvtColor');

  // 5. gaussianBlur
  const blurred = await src.gaussianBlur(3, 3, 0);
  assert.strictEqual(blurred.rows, 8);
  console.log('  ✓ gaussianBlur');

  // 6. resize
  const resized = await src.resize(16, 16);
  assert.strictEqual(resized.rows, 16);
  assert.strictEqual(resized.cols, 16);
  console.log('  ✓ resize');

  // 7. Constants present
  assert.ok(typeof cv.CV_8U === 'number', 'CV_8U');
  assert.ok(typeof cv.ColorCode === 'object', 'ColorCode enum');
  assert.ok(typeof cv.TemplateMatchMode === 'object', 'TemplateMatchMode enum');
  console.log('  ✓ Constants / enums');

  // 8. getBuildInformation
  assert.ok(typeof cv.getBuildInformation === 'function', 'getBuildInformation fn');
  const info = cv.getBuildInformation();
  assert.ok(typeof info === 'string' && info.includes('OpenCV'), 'OpenCV in build info');
  const firstLine = info.trim().split('\n')[0];
  console.log('  ✓ getBuildInformation:', firstLine);

  console.log('\nAll checks passed ✓');
}

main().catch(err => {
  console.error('\nTest failed:', err.message);
  process.exit(1);
});
