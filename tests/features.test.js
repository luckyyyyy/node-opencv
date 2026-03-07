'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const cv = require('../index.js');
const path = require('node:path');
const fs = require('node:fs');

const FIXTURES = path.join(__dirname, 'fixtures');
const LENA_JPG = path.join(FIXTURES, 'lena.jpg');

describe('Features 特征检测模块', () => {
  test('findContours 查找轮廓', async () => {
    // 创建带有白色矩形的黑色图像
    const mat = cv.Mat.zeros(100, 100, cv.CV_8UC1);
    // 手动设置中央矩形区域为白色
    const data = mat.data;
    // 在 [20,20]-[80,80] 区域设置白色（通过重构buffer）
    const buf = Buffer.alloc(100 * 100, 0);
    for (let y = 20; y < 80; y++) {
      for (let x = 20; x < 80; x++) {
        buf[y * 100 + x] = 255;
      }
    }
    const filledMat = cv.Mat.fromBuffer(100, 100, cv.CV_8UC1, buf);
    const contours = await filledMat.findContours(cv.ContourRetrievalMode.External, cv.ContourApproximation.Simple);
    assert.ok(Array.isArray(contours));
    assert.ok(contours.length >= 1);
    assert.ok(contours[0].length >= 4);
  });

  test('boundingRect 轮廓包围矩形', async () => {
    const contour = [
      { x: 10, y: 10 },
      { x: 50, y: 10 },
      { x: 50, y: 50 },
      { x: 10, y: 50 },
    ];
    const rect = cv.boundingRect(contour);
    assert.equal(rect.x, 10);
    assert.equal(rect.y, 10);
    assert.equal(rect.width, 41); // 50 - 10 + 1
    assert.equal(rect.height, 41);
  });

  test('getStructuringElement 获取结构元素', async () => {
    const kernel = cv.getStructuringElement(cv.MorphShape.Rect, 5, 5);
    assert.equal(kernel.rows, 5);
    assert.equal(kernel.cols, 5);
    // Rect 类型的 kernel 所有 25 个元素应全为 1
    for (const v of kernel.data) assert.equal(v, 1, `Rect kernel 元素应为 1，实际得到 ${v}`);
  });

  test('getStructuringElement MORPH_ELLIPSE', async () => {
    const kernel = cv.getStructuringElement(cv.MorphShape.Ellipse, 7, 7);
    assert.equal(kernel.rows, 7);
    assert.equal(kernel.cols, 7);
    // 中心像素应为 1
    assert.equal(kernel.data[3 * 7 + 3], 1, '源圆 kernel 中心应为 1');
    // 角落 (0,0) 在源圆外，应为 0
    assert.equal(kernel.data[0], 0, '源圆 kernel 角落应为 0');
  });

  test('getStructuringElement MORPH_CROSS', async () => {
    const kernel = cv.getStructuringElement(cv.MorphShape.Cross, 5, 5);
    assert.equal(kernel.rows, 5);
    assert.equal(kernel.cols, 5);
    // 十字模式：中心行和中心列应为 1，角落应为 0
    assert.equal(kernel.data[2 * 5 + 2], 1, '中心应为 1');
    assert.equal(kernel.data[2 * 5 + 0], 1, '中心行最左应为 1');
    assert.equal(kernel.data[0 * 5 + 2], 1, '中心列最上应为 1');
    assert.equal(kernel.data[0], 0, '角落 (0,0) 应为 0');
    assert.equal(kernel.data[4], 0, '角落 (0,4) 应为 0');
  });

  test('goodFeaturesToTrack 真实图像特征点检测', async () => {
    // 用 lena.jpg 灰度图检测角点，比合成梯度更贴近真实使用场景
    const lena = await cv.imread(LENA_JPG);
    const gray = await lena.cvtColor(cv.ColorCode.Bgr2Gray);
    const corners = await gray.goodFeaturesToTrack(50, 0.01, 10.0);
    assert.ok(Array.isArray(corners));
    // lena 是自然图像，应能检测到足够多的角点
    assert.ok(corners.length > 0, `应检测到特征点，实际 ${corners.length}`);
    assert.ok(corners.length <= 50, `最多返回 50 个，实际 ${corners.length}`);
    for (const pt of corners) {
      assert.equal(typeof pt.x, 'number');
      assert.equal(typeof pt.y, 'number');
      assert.ok(pt.x >= 0 && pt.x < 512);
      assert.ok(pt.y >= 0 && pt.y < 512);
    }
  });

  test('houghCircles 圆形检测', async () => {
    // 在 200x200 图上绘制圆心(100,100)、半径4的圆形轮廓
    const buf = Buffer.alloc(200 * 200, 0);
    const cx = 100, cy = 100, r = 40;
    for (let angle = 0; angle < 360; angle++) {
      const rad = angle * Math.PI / 180;
      const x = Math.round(cx + r * Math.cos(rad));
      const y = Math.round(cy + r * Math.sin(rad));
      if (x >= 0 && x < 200 && y >= 0 && y < 200) buf[y * 200 + x] = 255;
    }
    const circleMat = cv.Mat.fromBuffer(200, 200, cv.CV_8UC1, buf);
    // 高斯模糊提升检测稳定性
    const blurred = await circleMat.gaussianBlur(5, 5, 1.5);
    const circles = await blurred.houghCircles(1.0, 50, 100, 20, 30, 55);
    assert.ok(Array.isArray(circles));
    assert.ok(circles.length >= 1, `应至少检测到 1 个圆，实际得到 ${circles.length} 个`);
    // 检测到的圆心应接近 (100, 100)，半径接近 40
    const c = circles[0];
    assert.ok(typeof c.x === 'number' && typeof c.y === 'number' && typeof c.radius === 'number');
    assert.ok(Math.abs(c.x - cx) < 15, `圆心 cx=${c.x} 应接近 ${cx}`);
    assert.ok(Math.abs(c.y - cy) < 15, `圆心 cy=${c.y} 应接近 ${cy}`);
    assert.ok(Math.abs(c.radius - r) < 15, `半径=${c.radius} 应接近 ${r}`);
  });
});
