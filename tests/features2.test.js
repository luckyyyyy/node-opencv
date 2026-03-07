'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const cv = require('../index.js');

function makeSquareContour(x, y, size) {
  return [
    { x: x,        y: y },
    { x: x + size, y: y },
    { x: x + size, y: y + size },
    { x: x,        y: y + size },
  ];
}

describe('Features 新增操作', () => {
  test('arcLength 轮廓周长', async () => {
    // 边长为10的正方形周长 = 40
    const contour = makeSquareContour(0, 0, 10);
    const len = cv.arcLength(contour, true);
    assert.ok(len > 0);
    assert.ok(Math.abs(len - 40) < 2);
  });

  test('approxPolyDP 多边形近似', async () => {
    // 正方形轮廓近似后应保留约4个顶点
    const contour = makeSquareContour(10, 10, 50);
    const approx = cv.approxPolyDP(contour, 3.0, true);
    assert.ok(Array.isArray(approx));
    assert.ok(approx.length >= 3);
    approx.forEach(p => {
      assert.ok(typeof p.x === 'number');
      assert.ok(typeof p.y === 'number');
    });
  });

  test('convexHull 凸包', async () => {
    const points = [
      { x: 10, y: 10 },
      { x: 50, y: 10 },
      { x: 50, y: 50 },
      { x: 10, y: 50 },
      { x: 30, y: 30 }, // interior point
    ];
    const hull = cv.convexHull(points);
    assert.ok(Array.isArray(hull));
    // 凸包应包含4个顶点（内部点被排除）
    assert.equal(hull.length, 4);
  });

  test('moments 矩', async () => {
    const buf = Buffer.alloc(100 * 100, 0);
    for (let y = 30; y < 70; y++) {
      for (let x = 30; x < 70; x++) {
        buf[y * 100 + x] = 255;
      }
    }
    const mat = cv.Mat.fromBuffer(100, 100, cv.CV_8UC1, buf);
    const m = await mat.moments();
    assert.ok(m.m00 > 0);      // 面积
    assert.ok(typeof m.m10 === 'number');
    assert.ok(typeof m.m01 === 'number');
    assert.ok(typeof m.mu20 === 'number');
    // 质心 cx = m10/m00, cy = m01/m00 ≈ 49.5
    const cx = m.m10 / m.m00;
    const cy = m.m01 / m.m00;
    assert.ok(Math.abs(cx - 49.5) < 1);
    assert.ok(Math.abs(cy - 49.5) < 1);
  });

  test('drawContours 绘制轮廓', () => {
    const mat = cv.Mat.zeros(100, 100, cv.CV_8UC3);
    const contour = makeSquareContour(20, 20, 60);
    cv.drawContours(mat, [contour], 0, { v0: 0, v1: 255, v2: 0, v3: 0 }, 1);
    assert.equal(mat.rows, 100);
    assert.equal(mat.cols, 100);
    assert.equal(mat.channels, 3);
    // 绘制的轮廓左上角 (20,20) 应为绿色
    const idx = (20 * 100 + 20) * 3;
    assert.equal(mat.data[idx + 1], 255, 'G 通道应为 255');
    assert.equal(mat.data[idx],     0,   'B 通道应为 0');
    assert.equal(mat.data[idx + 2], 0,   'R 通道应为 0');
  });
});
