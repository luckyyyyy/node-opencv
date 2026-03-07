'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const cv = require('../index.js');

describe('Drawing 绘图模块', () => {
  test('drawLine 绘制线条', () => {
    const mat = cv.Mat.zeros(100, 100, cv.CV_8UC3);
    cv.drawLine(
      mat,
      { x: 0, y: 0 },
      { x: 99, y: 99 },
      { v0: 0, v1: 255, v2: 0, v3: 0 },
      2
    );
    assert.equal(mat.rows, 100);
    assert.equal(mat.cols, 100);
    // 确认有非零像素 (对角线上的绿色)
    const data = mat.data;
    let hasGreen = false;
    for (let i = 1; i < data.length; i += 3) {
      if (data[i] > 0) { hasGreen = true; break; }
    }
    assert.ok(hasGreen);
  });

  test('drawRectangle 绘制矩形', () => {
    const mat = cv.Mat.zeros(100, 100, cv.CV_8UC3);
    cv.drawRectangle(
      mat,
      { x: 10, y: 10, width: 50, height: 50 },
      { v0: 255, v1: 0, v2: 0, v3: 0 }
    );
    assert.equal(mat.rows, 100);
    assert.equal(mat.cols, 100);
    // 矩形左上角 (10,10) 应为蓝色 (v0=255, v1=0, v2=0)
    const idx = (10 * 100 + 10) * 3;
    assert.equal(mat.data[idx],     255, 'B 通道应为 255');
    assert.equal(mat.data[idx + 1],   0, 'G 通道应为 0');
    assert.equal(mat.data[idx + 2],   0, 'R 通道应为 0');
  });

  test('drawCircle 绘制圆形', () => {
    const mat = cv.Mat.zeros(100, 100, cv.CV_8UC3);
    cv.drawCircle(
      mat,
      { x: 50, y: 50 },
      30,
      { v0: 0, v1: 0, v2: 255, v3: 0 }
    );
    assert.equal(mat.rows, 100);
    assert.equal(mat.cols, 100);
  });

  test('drawCircle 填充 (thickness=-1)', () => {
    const mat = cv.Mat.zeros(100, 100, cv.CV_8UC3);
    cv.drawCircle(
      mat,
      { x: 50, y: 50 },
      20,
      { v0: 255, v1: 255, v2: 0, v3: 0 },
      -1
    );
    // 中心像素(50,50)应有颜色
    const idx = (50 * 100 + 50) * 3;
    assert.equal(mat.data[idx], 255); // Blue
  });

  test('putText 写入文字', () => {
    const mat = cv.Mat.zeros(100, 300, cv.CV_8UC3);
    cv.putText(
      mat,
      'Hello',
      { x: 10, y: 50 },
      cv.HersheyFont.Simplex,
      1.0,
      { v0: 255, v1: 255, v2: 255, v3: 0 }
    );
    assert.equal(mat.rows, 100);
    assert.equal(mat.cols, 300);
    // 确认有白色像素
    const data = mat.data;
    let hasWhite = false;
    for (let i = 0; i < data.length; i += 3) {
      if (data[i] === 255 && data[i+1] === 255 && data[i+2] === 255) {
        hasWhite = true; break;
      }
    }
    assert.ok(hasWhite);
  });

  test('drawEllipse 绘制椭圆', () => {
    const mat = cv.Mat.zeros(100, 100, cv.CV_8UC3);
    cv.drawEllipse(
      mat,
      { x: 50, y: 50 },
      { width: 30, height: 20 },
      0, 0, 360,
      { v0: 0, v1: 255, v2: 0, v3: 0 }
    );
    assert.equal(mat.rows, 100);
    assert.equal(mat.cols, 100);
  });

  test('getTextSize 获取文字尺寸', () => {
    const size = cv.getTextSize('Test', cv.HersheyFont.Simplex, 1.0);
    assert.ok(size.width > 0);
    assert.ok(size.height > 0);
    assert.equal(typeof size.baseline, 'number');
  });

  test('fillPoly 填充多边形', () => {
    const mat = cv.Mat.zeros(100, 100, cv.CV_8UC3);
    const pts = [[
      { x: 10, y: 10 },
      { x: 90, y: 10 },
      { x: 50, y: 90 },
    ]];
    cv.fillPoly(
      mat,
      pts,
      { v0: 0, v1: 255, v2: 0, v3: 0 }
    );
    assert.equal(mat.rows, 100);
    assert.equal(mat.cols, 100);
    // 三角形内心附近 (重心≈(50,37)) 应为绿色
    const idx = (37 * 100 + 50) * 3;
    assert.equal(mat.data[idx + 1], 255, '内部 G 通道应为 255');
    assert.equal(mat.data[idx],       0, '内部 B 通道应为 0');
  });

  test('drawLine 原地修改 Mat', () => {
    const mat = cv.Mat.zeros(50, 50, cv.CV_8UC3);
    cv.drawLine(
      mat,
      { x: 0, y: 0 },
      { x: 49, y: 49 },
      { v0: 255, v1: 0, v2: 0, v3: 0 }
    );
    // 原地绘制 — mat 本身应已被修改（对角线上第 0 像素 b=255）
    assert.equal(mat.data[0], 255);
  });
});
