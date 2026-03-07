'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const cv = require('../index.js');

const PI = Math.PI;

// Build a grayscale image with a large white rectangle on black
function rectMat(rows, cols, rx, ry, rw, rh) {
  const buf = Buffer.alloc(rows * cols, 0);
  for (let r = ry; r < ry + rh; r++)
    for (let c = rx; c < rx + rw; c++)
      buf[r * cols + c] = 255;
  return cv.Mat.fromBuffer(rows, cols, cv.CV_8UC1, buf);
}

describe('Features 补充+新增测试', () => {
  test('contourArea 轮廓面积', async () => {
    // 矩形轮廓面积 = 40*40 = 1600
    const contour = [
      { x: 0, y: 0 }, { x: 40, y: 0 },
      { x: 40, y: 40 }, { x: 0, y: 40 },
    ];
    const area = cv.contourArea(contour);
    assert.ok(Math.abs(area - 1600) < 5);
  });

  test('minAreaRect 最小外接矩形', async () => {
    const contour = [
      { x: 10, y: 10 }, { x: 60, y: 10 },
      { x: 60, y: 40 }, { x: 10, y: 40 },
    ];
    const r = cv.minAreaRect(contour);
    assert.ok(typeof r.center.x === 'number');
    assert.ok(typeof r.size.width === 'number');
    assert.ok(typeof r.angle === 'number');
    // 宽度或高度约 50
    const longer = Math.max(r.size.width, r.size.height);
    assert.ok(Math.abs(longer - 50) < 2);
  });

  test('houghLines 标准霍夫直线', async () => {
    // 在100x100图像上绘制一条水平线
    const mat = rectMat(100, 100, 0, 50, 100, 2);
    const edges = await mat.canny(50, 150);
    const lines = await edges.houghLines(1, PI / 180, 50);
    assert.ok(Array.isArray(lines));
    // 至少检测到一条线
    assert.ok(lines.length >= 1);
    assert.ok(typeof lines[0].rho === 'number');
    assert.ok(typeof lines[0].theta === 'number');
  });

  test('houghLinesP 概率霍夫直线', async () => {
    const mat = rectMat(100, 100, 0, 50, 100, 2);
    const edges = await mat.canny(50, 150);
    const segs = await edges.houghLinesP(1, PI / 180, 30, 20.0, 5.0);
    assert.ok(Array.isArray(segs));
    segs.forEach(s => {
      assert.ok(typeof s.x1 === 'number');
      assert.ok(typeof s.y1 === 'number');
      assert.ok(typeof s.x2 === 'number');
      assert.ok(typeof s.y2 === 'number');
    });
  });

  test('adaptiveThreshold 自适应阈值', async () => {
    const buf = Buffer.alloc(50 * 50);
    for (let i = 0; i < 50 * 50; i++) buf[i] = (i % 128);
    const mat = cv.Mat.fromBuffer(50, 50, cv.CV_8UC1, buf);
    const out = await mat.adaptiveThreshold(
      255,
      cv.AdaptiveThresholdType.MeanC,
      cv.ThresholdType.Binary,
      11, 2,
    );
    assert.equal(out.rows, 50);
    assert.equal(out.cols, 50);
    // output should be 0 or 255 only
    const unique = new Set(out.data);
    assert.ok(unique.size <= 2);
  });

  test('adaptiveThreshold GAUSSIAN_C 方法', async () => {
    const buf = Buffer.alloc(30 * 30, 128);
    const mat = cv.Mat.fromBuffer(30, 30, cv.CV_8UC1, buf);
    const out = await mat.adaptiveThreshold(
      255,
      cv.AdaptiveThresholdType.GaussianC,
      cv.ThresholdType.Binary,
      11, 5,
    );
    assert.equal(out.rows, 30);
    assert.equal(out.cols, 30);
  });

  test('getPerspectiveTransform 透视变换矩阵', async () => {
    const src = [
      { x: 0, y: 0 }, { x: 100, y: 0 },
      { x: 100, y: 100 }, { x: 0, y: 100 },
    ];
    const dst = [
      { x: 10, y: 10 }, { x: 90, y: 10 },
      { x: 90, y: 90 }, { x: 10, y: 90 },
    ];
    const M = await cv.getPerspectiveTransform(src, dst);
    assert.equal(M.rows, 3);
    assert.equal(M.cols, 3);
  });

  test('getAffineTransform 仿射变换矩阵', async () => {
    const src = [{ x: 0, y: 0 }, { x: 100, y: 0 }, { x: 0, y: 100 }];
    const dst = [{ x: 10, y: 10 }, { x: 110, y: 10 }, { x: 10, y: 110 }];
    const M = await cv.getAffineTransform(src, dst);
    assert.equal(M.rows, 2);
    assert.equal(M.cols, 3);
  });

  test('getPerspectiveTransform 点数不足应抛出错误', async () => {
    await assert.rejects(
      () => cv.getPerspectiveTransform([{ x: 0, y: 0 }], [{ x: 0, y: 0 }]),
      /Expected exactly 4 points/i,
    );
  });

  test('getAffineTransform 点数不足应抛出错误', async () => {
    await assert.rejects(
      () => cv.getAffineTransform([{ x: 0, y: 0 }], [{ x: 0, y: 0 }]),
      /Expected exactly 3 points/i,
    );
  });
});
