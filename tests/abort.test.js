'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const os = require('node:os');
const cv = require('../index.js');

const g = (r, c, v = 128) => cv.Mat.fromBuffer(r, c, cv.CV_8UC1, Buffer.alloc(r * c, v));
const g3 = (r, c, v = 0) => cv.Mat.fromBuffer(r, c, cv.CV_8UC3, Buffer.alloc(r * c * 3, v));
const sig = () => new AbortController().signal;
const aborted = () => { const ac = new AbortController(); ac.abort(); return ac.signal; };

async function acceptAbortOrResult(promise) {
  try { await promise; } catch (e) { assert.ok(e instanceof Error, `应抛出 Error，实际: ${e}`); }
}

// ─── AbortSignal 正向参数覆盖测试（表格驱动）────────────────────────────────
// 每条 [名称, async (signal) => void]
describe('AbortSignal 正向参数覆盖测试', () => {
  const cases = [
    ['Mat.resize',          async s => assert.equal((await g(20,20).resize(10,10,null,null,null,s)).rows, 10)],
    ['Mat.gaussianBlur',    async s => assert.equal((await g(10,10).gaussianBlur(3,3,1.0,null,s)).rows, 10)],
    ['Mat.cvtColor',        async s => assert.equal((await g3(4,4).cvtColor(cv.ColorCode.Bgr2Gray,s)).channels, 1)],
    ['Mat.threshold',       async s => assert.equal((await g(4,4,100).threshold(127,255,cv.ThresholdType.Binary,s)).rows, 4)],
    ['Mat.canny',           async s => assert.equal((await g(10,10).canny(50,150,null,null,s)).rows, 10)],
    ['Mat.flip',            async s => assert.equal((await cv.Mat.fromBuffer(2,2,cv.CV_8UC1,Buffer.from([1,2,3,4])).flip(cv.FlipCode.Horizontal,s)).data[0], 2)],
    ['Mat.crop',            async s => assert.equal((await g(20,20).crop(0,0,10,10,s)).rows, 10)],
    ['Mat.minMaxLoc',       async s => assert.equal(typeof (await g(4,4,50).minMaxLoc(s)).minVal, 'number')],
    ['Mat.split',           async s => assert.equal((await g3(4,4).split(s)).length, 3)],
    ['Mat.normalize',       async s => assert.equal((await g(4,4,100).normalize(0,255,cv.NormType.MinMax,null,s)).rows, 4)],
    ['Mat.addWeighted',     async s => assert.equal((await g(4,4,100).addWeighted(0.5,g(4,4,50),0.5,0,null,s)).rows, 4)],
    ['Mat.medianBlur',      async s => assert.equal((await g(9,9).medianBlur(3,s)).rows, 9)],
    ['Mat.bilateralFilter', async s => assert.equal((await g(10,10).bilateralFilter(5,75,75,s)).rows, 10)],
    ['Mat.equalizeHist',    async s => assert.equal((await g(10,10).equalizeHist(s)).rows, 10)],
    ['Mat.sobel',           async s => assert.equal((await g(10,10).sobel(cv.CV_16S,1,0,null,null,null,s)).rows, 10)],
    ['Mat.laplacian',       async s => assert.equal((await g(10,10).laplacian(cv.CV_16S,null,null,null,s)).rows, 10)],
    ['Mat.copyMakeBorder',  async s => assert.equal((await g(4,4).copyMakeBorder(1,1,1,1,cv.BorderType.Constant,null,s)).rows, 6)],
    ['Mat.absDiff',         async s => assert.equal((await g(4,4,100).absDiff(g(4,4,50),s)).data[0], 50)],
    ['Mat.add',             async s => assert.equal((await g(4,4,10).add(g(4,4,20),s)).data[0], 30)],
    ['Mat.subtract',        async s => assert.equal((await g(4,4,100).subtract(g(4,4,10),s)).data[0], 90)],
    ['Mat.multiply',        async s => assert.equal((await g(4,4,2).multiply(g(4,4,3),null,s)).data[0], 6)],
    ['Mat.bitwiseAnd',      async s => assert.equal((await cv.Mat.fromBuffer(1,1,cv.CV_8UC1,Buffer.from([0b11110000])).bitwiseAnd(cv.Mat.fromBuffer(1,1,cv.CV_8UC1,Buffer.from([0b11001100])),null,s)).data[0], 0b11000000)],
    ['Mat.bitwiseOr',       async s => assert.equal((await cv.Mat.fromBuffer(1,1,cv.CV_8UC1,Buffer.from([0b11000000])).bitwiseOr(cv.Mat.fromBuffer(1,1,cv.CV_8UC1,Buffer.from([0b00000011])),null,s)).data[0], 0b11000011)],
    ['Mat.bitwiseNot',      async s => assert.equal((await cv.Mat.fromBuffer(1,1,cv.CV_8UC1,Buffer.from([0])).bitwiseNot(null,s)).data[0], 255)],
    ['Mat.inRange',         async s => assert.equal((await g(4,4,100).inRange({v0:50,v1:0,v2:0,v3:0},{v0:150,v1:0,v2:0,v3:0},s)).data[0], 255)],
    ['Mat.matchTemplate',   async s => assert.ok(!(await g(20,20).matchTemplate(g(5,5),cv.TemplateMatchMode.Sqdiff,s)).empty)],
    ['Mat.matchTemplateAll',async s => assert.ok(Array.isArray(await g(20,20).matchTemplateAll(g(5,5),cv.TemplateMatchMode.Sqdiff,0.5,0.3,s)))],
    ['hconcat',             async s => assert.equal((await cv.hconcat([g(3,3),g(3,3)],s)).cols, 6)],
    ['vconcat',             async s => assert.equal((await cv.vconcat([g(3,3),g(3,3)],s)).rows, 6)],
    ['merge',               async s => assert.equal((await cv.merge([g(3,3,10),g(3,3,20),g(3,3,30)],s)).channels, 3)],
    ['nmsBoxes',            async s => assert.ok(Array.isArray(await cv.nmsBoxes([{x:0,y:0,width:100,height:100}],[0.9],0.5,0.5,s)))],
    ['getPerspectiveTransform', async s => assert.equal((await cv.getPerspectiveTransform(
      [{x:0,y:0},{x:100,y:0},{x:100,y:100},{x:0,y:100}],
      [{x:10,y:10},{x:90,y:10},{x:90,y:90},{x:10,y:90}], s)).rows, 3)],
    ['getAffineTransform',  async s => assert.equal((await cv.getAffineTransform(
      [{x:0,y:0},{x:100,y:0},{x:0,y:100}],
      [{x:10,y:10},{x:110,y:10},{x:10,y:110}], s)).rows, 2)],
  ];

  for (const [name, run] of cases) {
    test(`${name} 接受 AbortSignal 参数`, async () => run(sig()));
  }

  // 需要多行 setup 的用例
  test('Mat.transpose 接受 AbortSignal 参数', async () => {
    const mat = cv.Mat.fromBuffer(2, 3, cv.CV_8UC1, Buffer.from([1, 2, 3, 4, 5, 6]));
    const out = await mat.transpose(sig());
    assert.equal(out.rows, 3); assert.equal(out.cols, 2);
  });

  test('Mat.warpAffine 接受 AbortSignal 参数', async () => {
    const M = cv.Mat.fromBuffer(2, 3, cv.CV_64F, Buffer.from(new Float64Array([1, 0, 0, 0, 1, 0]).buffer));
    assert.equal((await g3(10, 10, 100).warpAffine(M, 10, 10, null, null, sig())).rows, 10);
  });

  test('Mat.warpPerspective 接受 AbortSignal 参数', async () => {
    const M = cv.Mat.fromBuffer(3, 3, cv.CV_64F, Buffer.from(new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]).buffer));
    assert.equal((await g3(10, 10, 100).warpPerspective(M, 10, 10, null, null, sig())).rows, 10);
  });

  test('Mat.filter2D 接受 AbortSignal 参数', async () => {
    const k = cv.Mat.fromBuffer(3, 3, cv.CV_32F, Buffer.from(new Float32Array(9).fill(1 / 9).buffer));
    assert.equal((await g(10, 10, 100).filter2D(-1, k, null, sig())).rows, 10);
  });

  test('Mat.dilate 接受 AbortSignal 参数', async () => {
    assert.equal((await g(10,10).dilate(cv.getStructuringElement(cv.MorphShape.Rect,3,3), null, sig())).rows, 10);
  });

  test('Mat.erode 接受 AbortSignal 参数', async () => {
    assert.equal((await g(10,10).erode(cv.getStructuringElement(cv.MorphShape.Rect,3,3), null, sig())).rows, 10);
  });

  test('Mat.morphologyEx 接受 AbortSignal 参数', async () => {
    assert.equal((await g(10,10).morphologyEx(cv.MorphType.Open, cv.getStructuringElement(cv.MorphShape.Rect,3,3), null, sig())).rows, 10);
  });

  test('imdecode 接受 AbortSignal 参数', async () => {
    const buf = await cv.imencode('.png', g3(4, 4));
    assert.equal((await cv.imdecode(buf, null, sig())).rows, 4);
  });

  test('imencode 接受 AbortSignal 参数', async () => {
    assert.ok((await cv.imencode('.png', g3(4, 4), sig())).length > 0);
  });

  test('imwrite 接受 AbortSignal 参数', async () => {
    assert.equal(await cv.imwrite(path.join(os.tmpdir(), 'aborttest_fresh.png'), g3(4, 4), sig()), true);
  });

  test('blobFromImage 接受 AbortSignal 参数', async () => {
    assert.ok(!(await cv.blobFromImage(g3(4, 4, 128), 1.0, 4, 4, null, false, false, sig())).empty);
  });

  // 容错测试：传入已 abort 信号，成功或被拒绝都合法
  test('已 abort 信号传入 Mat.resize 不崩溃', async () => {
    await acceptAbortOrResult(g(10, 10).resize(5, 5, null, null, null, aborted()));
  });

  test('已 abort 信号传入 imencode 不崩溃', async () => {
    await acceptAbortOrResult(cv.imencode('.png', g3(4, 4), aborted()));
  });

  test('已 abort 信号传入 blobFromImage 不崩溃', async () => {
    await acceptAbortOrResult(cv.blobFromImage(g3(4, 4, 128), 1.0, 4, 4, null, false, false, aborted()));
  });
});

// ─── Features 模块 AbortSignal 覆盖 ───────────────────────────────────────────
describe('Features AbortSignal 参数覆盖', () => {
  test('Mat.findContours 接受 AbortSignal 参数', async () => {
    const buf = Buffer.alloc(20 * 20, 0);
    for (let y = 5; y < 15; y++) for (let x = 5; x < 15; x++) buf[y * 20 + x] = 255;
    const mat = cv.Mat.fromBuffer(20, 20, cv.CV_8UC1, buf);
    const contours = await mat.findContours(
      cv.ContourRetrievalMode.External,
      cv.ContourApproximation.Simple,
      sig(),
    );
    assert.ok(contours.length >= 1);
  });

  test('Mat.goodFeaturesToTrack 接受 AbortSignal 参数', async () => {
    const buf = Buffer.alloc(30 * 30, 0);
    for (let y = 0; y < 30; y++) for (let x = 0; x < 30; x++) buf[y * 30 + x] = ((x + y) % 10) * 20;
    const mat = cv.Mat.fromBuffer(30, 30, cv.CV_8UC1, buf);
    const pts = await mat.goodFeaturesToTrack(10, 0.01, 3.0, null, null, null, null, sig());
    assert.ok(Array.isArray(pts));
  });

  test('Mat.houghCircles 接受 AbortSignal 参数', async () => {
    const buf = Buffer.alloc(100 * 100, 0);
    for (let a = 0; a < 360; a++) {
      const x = Math.round(50 + 20 * Math.cos(a * Math.PI / 180));
      const y = Math.round(50 + 20 * Math.sin(a * Math.PI / 180));
      if (x >= 0 && x < 100 && y >= 0 && y < 100) buf[y * 100 + x] = 255;
    }
    const mat = cv.Mat.fromBuffer(100, 100, cv.CV_8UC1, buf);
    const blurred = await mat.gaussianBlur(5, 5, 1.5);
    const circles = await blurred.houghCircles(1.0, 30, 100, 20, 10, 30, null, sig());
    assert.ok(Array.isArray(circles));
  });

  test('Mat.houghLines 接受 AbortSignal 参数', async () => {
    const buf = Buffer.alloc(50 * 50, 0);
    for (let c = 0; c < 50; c++) { buf[25 * 50 + c] = 255; buf[26 * 50 + c] = 255; }
    const mat = cv.Mat.fromBuffer(50, 50, cv.CV_8UC1, buf);
    const edges = await mat.canny(50, 150);
    const lines = await edges.houghLines(1, Math.PI / 180, 20, sig());
    assert.ok(Array.isArray(lines));
  });

  test('Mat.houghLinesP 接受 AbortSignal 参数', async () => {
    const buf = Buffer.alloc(50 * 50, 0);
    for (let c = 0; c < 50; c++) { buf[25 * 50 + c] = 255; }
    const mat = cv.Mat.fromBuffer(50, 50, cv.CV_8UC1, buf);
    const edges = await mat.canny(50, 150);
    const segs = await edges.houghLinesP(1, Math.PI / 180, 20, 10.0, 5.0, sig());
    assert.ok(Array.isArray(segs));
  });

  test('Mat.adaptiveThreshold 接受 AbortSignal 参数', async () => {
    const buf = Buffer.alloc(30 * 30);
    for (let i = 0; i < 900; i++) buf[i] = i % 128;
    const mat = cv.Mat.fromBuffer(30, 30, cv.CV_8UC1, buf);
    const out = await mat.adaptiveThreshold(
      255, cv.AdaptiveThresholdType.MeanC, cv.ThresholdType.Binary, 11, 2, sig(),
    );
    assert.equal(out.rows, 30);
  });
});

// ─── Video AbortSignal 覆盖 ────────────────────────────────────────────────────
describe('Video AbortSignal 参数覆盖', () => {
  test('VideoWriter.write 接受 AbortSignal 参数', async () => {
    const fs = require('node:fs');
    const tmpFile = path.join(os.tmpdir(), `abort_vid_${Date.now()}.avi`);
    const writer = cv.VideoWriter.open(tmpFile, 'MJPG', 10, 10, 10, true);
    await writer.write(g3(10, 10), sig());
    writer.release();
    if (fs.existsSync(tmpFile)) fs.unlinkSync(tmpFile);
  });
});

// ─── Net AbortSignal 覆盖 ──────────────────────────────────────────────────────
describe('Net AbortSignal 参数覆盖', () => {
  const ONNX_MODEL = path.join(__dirname, 'fixtures', 'identity.onnx');

  test('Net.run 接受 AbortSignal 参数', async () => {
    const net = cv.Net.readNetFromOnnx(ONNX_MODEL);
    const data = new Float32Array(16).fill(0.5);
    const img = cv.Mat.fromBuffer(4, 4, cv.CV_32FC1, Buffer.from(data.buffer));
    const blob = await cv.blobFromImage(img, 1.0, 4, 4);
    const out = await net.run(blob, 'input', 'output', sig());
    assert.ok(!out.empty);
  });

  test('Net.runMultiple 接受 AbortSignal 参数', async () => {
    const net = cv.Net.readNetFromOnnx(ONNX_MODEL);
    const data = new Float32Array(16).fill(0.25);
    const img = cv.Mat.fromBuffer(4, 4, cv.CV_32FC1, Buffer.from(data.buffer));
    const blob = await cv.blobFromImage(img, 1.0, 4, 4);
    const outs = await net.runMultiple(blob, 'input', ['output'], sig());
    assert.equal(outs.length, 1);
  });
});
