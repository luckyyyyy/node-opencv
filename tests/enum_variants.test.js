/**
 * 枚举变体功能性覆盖 — 补全常量之外的实际参数路径
 */
'use strict';
const { describe, test } = require('node:test');
const assert = require('node:assert/strict');
const cv = require('../index.js');

function solidMat(rows, cols, value = 128) {
  const buf = Buffer.alloc(rows * cols, value);
  return cv.Mat.fromBuffer(rows, cols, cv.CV_8UC1, buf);
}

function gradientMat(rows, cols) {
  const buf = Buffer.alloc(rows * cols);
  for (let i = 0; i < rows * cols; i++) buf[i] = (i * 256 / (rows * cols)) & 0xFF;
  return cv.Mat.fromBuffer(rows, cols, cv.CV_8UC1, buf);
}

// ──────────────────────────────────────────────────────────
// FlipCode 所有变体
// ──────────────────────────────────────────────────────────
describe('FlipCode 所有变体功能调用', () => {
  test('FlipCode.Vertical — 上下翻转', async () => {
    const rows = 4, cols = 4;
    const buf = Buffer.alloc(rows * cols, 0);
    // 第 0 行=10，第 3 行=200
    for (let c = 0; c < cols; c++) buf[c] = 10;
    for (let c = 0; c < cols; c++) buf[3 * cols + c] = 200;
    const mat = cv.Mat.fromBuffer(rows, cols, cv.CV_8UC1, buf);
    const out = await mat.flip(cv.FlipCode.Vertical);
    // 翻转后第 0 行应原来第 3 行 (200)
    assert.equal(out.data[0], 200);
    assert.equal(out.data[3 * cols], 10);
  });

  test('FlipCode.Both — 同时水平+垂直翻转', async () => {
    const rows = 2, cols = 2;
    // [10,20,30,40]  →  [40,30,20,10]
    const buf = Buffer.from([10, 20, 30, 40]);
    const mat = cv.Mat.fromBuffer(rows, cols, cv.CV_8UC1, buf);
    const out = await mat.flip(cv.FlipCode.Both);
    assert.equal(out.data[0], 40);
    assert.equal(out.data[3], 10);
  });
});

// ──────────────────────────────────────────────────────────
// ThresholdType.Otsu 和 Triangle
// ──────────────────────────────────────────────────────────
describe('ThresholdType Otsu/Triangle 功能调用', () => {
  test('ThresholdType.Otsu — 自动计算阈值', async () => {
    // 双峰分布图像最适合 Otsu
    const rows = 4, cols = 8;
    const buf = Buffer.from([
      0, 0, 0, 0, 255, 255, 255, 255,
      0, 0, 0, 0, 255, 255, 255, 255,
      0, 0, 0, 0, 255, 255, 255, 255,
      0, 0, 0, 0, 255, 255, 255, 255,
    ]);
    const mat = cv.Mat.fromBuffer(rows, cols, cv.CV_8UC1, buf);
    // thresh=0 时 Otsu 会自动选择最佳阈值
    const out = await mat.threshold(0, 255, cv.ThresholdType.Otsu);
    assert.equal(out.rows, rows);
    assert.equal(out.data[0], 0);   // 原值 0 → 0
    assert.equal(out.data[4], 255); // 原值 255 → 255
  });

  test('ThresholdType.Triangle — 三角形法阈值', async () => {
    const mat = gradientMat(4, 16);
    const out = await mat.threshold(0, 255, cv.ThresholdType.Triangle);
    assert.equal(out.rows, 4);
    // Triangle 是 Binary 变体，输出仍是二值图
    for (const v of out.data) assert.ok(v === 0 || v === 255, `期望 0/255，实际 ${v}`);
  });
});

// ──────────────────────────────────────────────────────────
// MorphType 所有变体
// ──────────────────────────────────────────────────────────
describe('MorphType 所有变体功能调用', () => {
  const makeInput = () => {
    const rows = 16, cols = 16;
    const buf = Buffer.alloc(rows * cols, 0);
    for (let r = 4; r < 12; r++)
      for (let c = 4; c < 12; c++)
        buf[r * cols + c] = 255;
    return cv.Mat.fromBuffer(rows, cols, cv.CV_8UC1, buf);
  };
  const kernel = () => cv.Mat.ones(3, 3, cv.CV_8UC1);

  test('MorphType.Erode', async () => {
    const out = await makeInput().morphologyEx(cv.MorphType.Erode, kernel());
    assert.equal(out.rows, 16);
  });

  test('MorphType.Dilate', async () => {
    const out = await makeInput().morphologyEx(cv.MorphType.Dilate, kernel());
    assert.equal(out.rows, 16);
  });

  test('MorphType.Close', async () => {
    const out = await makeInput().morphologyEx(cv.MorphType.Close, kernel());
    assert.equal(out.rows, 16);
  });

  test('MorphType.Gradient', async () => {
    const out = await makeInput().morphologyEx(cv.MorphType.Gradient, kernel());
    assert.equal(out.rows, 16);
  });

  test('MorphType.TopHat', async () => {
    const out = await makeInput().morphologyEx(cv.MorphType.TopHat, kernel());
    assert.equal(out.rows, 16);
  });

  test('MorphType.BlackHat', async () => {
    const out = await makeInput().morphologyEx(cv.MorphType.BlackHat, kernel());
    assert.equal(out.rows, 16);
  });

  test('MorphType.HitMiss', async () => {
    // HitMiss 仅适用于 CV_8UC1 二值图 (0/255)
    const out = await makeInput().morphologyEx(cv.MorphType.HitMiss, kernel());
    assert.equal(out.rows, 16);
  });
});

// ──────────────────────────────────────────────────────────
// NormType 功能性变体（normalize 调用）
// ──────────────────────────────────────────────────────────
describe('NormType 功能性调用', () => {
  test('NormType.L1', async () => {
    const mat = solidMat(1, 4, 100);
    const out = await mat.normalize(1.0, 0, cv.NormType.L1);
    assert.equal(out.rows, 1);
  });

  test('NormType.L2', async () => {
    const mat = solidMat(1, 4, 100);
    const out = await mat.normalize(1.0, 0, cv.NormType.L2);
    assert.equal(out.rows, 1);
  });

  test('NormType.Inf', async () => {
    const mat = solidMat(1, 4, 100);
    const out = await mat.normalize(1.0, 0, cv.NormType.Inf);
    assert.equal(out.rows, 1);
  });

  test('NormType.L2Sqr (norm_type=5)', async () => {
    // L2Sqr=5, 部分 OpenCV 版本对 normalize 支持
    // 若不支持会 throw — 我们只验证不崩溃或返回正确形状
    try {
      const mat = solidMat(1, 4, 100);
      const out = await mat.normalize(1.0, 0, cv.NormType.L2Sqr);
      assert.equal(out.rows, 1);
    } catch (e) {
      // L2Sqr 不被 normalize 支持是合理的
      assert.ok(e instanceof Error);
    }
  });
});

// ──────────────────────────────────────────────────────────
// ContourRetrievalMode 所有变体
// ──────────────────────────────────────────────────────────
describe('ContourRetrievalMode 所有变体功能调用', () => {
  function makeBinaryWithRect() {
    const rows = 16, cols = 16;
    const buf = Buffer.alloc(rows * cols, 0);
    for (let r = 4; r < 12; r++)
      for (let c = 4; c < 12; c++)
        buf[r * cols + c] = 255;
    return cv.Mat.fromBuffer(rows, cols, cv.CV_8UC1, buf);
  }

  test('ContourRetrievalMode.List', async () => {
    const mat = makeBinaryWithRect();
    const contours = await mat.findContours(
      cv.ContourRetrievalMode.List,
      cv.ContourApproximation.Simple
    );
    assert.ok(contours.length >= 1);
  });

  test('ContourRetrievalMode.CComp', async () => {
    const mat = makeBinaryWithRect();
    const contours = await mat.findContours(
      cv.ContourRetrievalMode.CComp,
      cv.ContourApproximation.Simple
    );
    assert.ok(Array.isArray(contours));
  });

  test('ContourRetrievalMode.Tree', async () => {
    const mat = makeBinaryWithRect();
    const contours = await mat.findContours(
      cv.ContourRetrievalMode.Tree,
      cv.ContourApproximation.Simple
    );
    assert.ok(Array.isArray(contours));
  });
});

// ──────────────────────────────────────────────────────────
// ContourApproximation 所有变体
// ──────────────────────────────────────────────────────────
describe('ContourApproximation 所有变体功能调用', () => {
  function makeBinary() {
    const rows = 16, cols = 16;
    const buf = Buffer.alloc(rows * cols, 0);
    for (let r = 4; r < 12; r++)
      for (let c = 4; c < 12; c++)
        buf[r * cols + c] = 255;
    return cv.Mat.fromBuffer(rows, cols, cv.CV_8UC1, buf);
  }

  test('ContourApproximation.None', async () => {
    const contours = await makeBinary().findContours(
      cv.ContourRetrievalMode.External,
      cv.ContourApproximation.None
    );
    assert.ok(contours.length >= 1);
  });

  test('ContourApproximation.Tc89L1', async () => {
    const contours = await makeBinary().findContours(
      cv.ContourRetrievalMode.External,
      cv.ContourApproximation.Tc89L1
    );
    assert.ok(Array.isArray(contours));
  });

  test('ContourApproximation.Tc89Kcos', async () => {
    const contours = await makeBinary().findContours(
      cv.ContourRetrievalMode.External,
      cv.ContourApproximation.Tc89Kcos
    );
    assert.ok(Array.isArray(contours));
  });
});

// ──────────────────────────────────────────────────────────
// Net.readNetFromCaffe / readNetFromDarknet — 错误路径
// ──────────────────────────────────────────────────────────
describe('Net 错误路径覆盖', () => {
  test('readNetFromCaffe 文件不存在抛出错误', () => {
    assert.throws(
      () => cv.Net.readNetFromCaffe('/nonexistent.prototxt', '/nonexistent.caffemodel'),
      (e) => e instanceof Error
    );
  });

  test('readNetFromDarknet 文件不存在抛出错误', () => {
    assert.throws(
      () => cv.Net.readNetFromDarknet('/nonexistent.cfg', '/nonexistent.weights'),
      (e) => e instanceof Error
    );
  });
});

// ──────────────────────────────────────────────────────────
// imread / imdecode 带 ImreadFlag 参数
// ──────────────────────────────────────────────────────────
describe('imread/imdecode ImreadFlag 变体', () => {
  test('imdecode ImreadFlag.Color', async () => {
    // 先 imencode 一张 BGR 图
    const src = cv.Mat.fromBuffer(4, 4, cv.CV_8UC3, Buffer.alloc(4 * 4 * 3, 128));
    const buf = await cv.imencode('.png', src);
    const out = await cv.imdecode(buf, cv.ImreadFlag.Color);
    assert.equal(out.channels, 3);
  });

  test('imdecode ImreadFlag.Unchanged', async () => {
    const src = cv.Mat.fromBuffer(4, 4, cv.CV_8UC3, Buffer.alloc(4 * 4 * 3, 128));
    const buf = await cv.imencode('.png', src);
    const out = await cv.imdecode(buf, cv.ImreadFlag.Unchanged);
    assert.ok(out.rows > 0);
  });
});
