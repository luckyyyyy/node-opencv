'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const cv = require('../index.js');

// ─── Mat 基础测试 ──────────────────────────────────────────────────────────────

describe('Mat 基础操作', () => {
  test('Mat 构造函数创建空矩阵', () => {
    const mat = new cv.Mat();
    assert.equal(mat.empty, true);
    assert.equal(mat.rows, 0);
    assert.equal(mat.cols, 0);
  });

  test('Mat.zeros 创建零矩阵', () => {
    const mat = cv.Mat.zeros(4, 4, cv.CV_8UC1);
    assert.equal(mat.rows, 4);
    assert.equal(mat.cols, 4);
    assert.equal(mat.channels, 1);
    assert.equal(mat.empty, false);
    const data = mat.data;
    assert.equal(data.length, 16);
    for (let i = 0; i < data.length; i++) {
      assert.equal(data[i], 0);
    }
  });

  test('Mat.ones 创建全1矩阵', () => {
    const mat = cv.Mat.ones(3, 3, cv.CV_8UC1);
    assert.equal(mat.rows, 3);
    assert.equal(mat.cols, 3);
    const data = mat.data;
    for (let i = 0; i < data.length; i++) {
      assert.equal(data[i], 1);
    }
  });

  test('Mat.fromBuffer 从 Buffer 创建矩阵', () => {
    const buf = Buffer.from([10, 20, 30, 40, 50, 60, 70, 80, 90]);
    const mat = cv.Mat.fromBuffer(3, 3, cv.CV_8UC1, buf);
    assert.equal(mat.rows, 3);
    assert.equal(mat.cols, 3);
    const data = mat.data;
    assert.equal(data[0], 10);
    assert.equal(data[4], 50);
    assert.equal(data[8], 90);
  });

  test('Mat.size 返回正确尺寸', () => {
    const mat = cv.Mat.zeros(5, 7, cv.CV_8UC1);
    const size = mat.size;
    assert.equal(size.width, 7);
    assert.equal(size.height, 5);
  });

  test('Mat.channels 返回正确通道数', () => {
    const mat3 = cv.Mat.zeros(2, 2, cv.CV_8UC3);
    assert.equal(mat3.channels, 3);
    const mat1 = cv.Mat.zeros(2, 2, cv.CV_8UC1);
    assert.equal(mat1.channels, 1);
  });

  test('Mat.total 返回元素总数', () => {
    const mat = cv.Mat.zeros(3, 4, cv.CV_8UC1);
    assert.equal(mat.total, 12);
  });

  test('Mat.elemSize 返回元素字节数', () => {
    const mat1 = cv.Mat.zeros(3, 4, cv.CV_8UC1);
    assert.equal(mat1.elemSize, 1);
    const mat3 = cv.Mat.zeros(3, 4, cv.CV_8UC3);
    assert.equal(mat3.elemSize, 3);
  });

  test('Mat.clone 克隆矩阵', () => {
    const buf = Buffer.from([1, 2, 3, 4]);
    const mat = cv.Mat.fromBuffer(2, 2, cv.CV_8UC1, buf);
    const cloned = mat.clone();
    const data = cloned.data;
    assert.equal(data[0], 1);
    assert.equal(data[3], 4);
    assert.equal(cloned.rows, 2);
  });

  test('Mat.convertTo 转换类型', async () => {
    const buf = Buffer.from([100, 200, 50]);
    const mat = cv.Mat.fromBuffer(1, 3, cv.CV_8UC1, buf);
    const f32mat = await mat.convertTo(cv.CV_32FC1, 1.0 / 255.0);
    assert.equal(f32mat.rows, 1);
    assert.equal(f32mat.cols, 3);
    assert.equal(f32mat.depth, cv.CV_32F);
    // 验证实际转换后的浮点值 (pixel / 255)
    const floats = new Float32Array(f32mat.data.buffer, f32mat.data.byteOffset, 3);
    assert.ok(Math.abs(floats[0] - 100 / 255) < 1e-4, `floats[0]=${floats[0]}`);
    assert.ok(Math.abs(floats[1] - 200 / 255) < 1e-4, `floats[1]=${floats[1]}`);
    assert.ok(Math.abs(floats[2] -  50 / 255) < 1e-4, `floats[2]=${floats[2]}`);
  });

  test('Mat.release 释放矩阵', () => {
    const mat = cv.Mat.zeros(5, 5, cv.CV_8UC3);
    assert.equal(mat.empty, false);
    mat.release();
    assert.equal(mat.empty, true);
  });
});

// ─── Mat 图像处理测试 ──────────────────────────────────────────────────────────

describe('Mat 图像处理 (async)', () => {
  test('Mat.threshold 二值化', async () => {
    const buf = Buffer.from([0, 50, 100, 150, 200, 255]);
    const mat = cv.Mat.fromBuffer(1, 6, cv.CV_8UC1, buf);
    const result = await mat.threshold(127, 255, cv.ThresholdType.Binary);
    const data = result.data;
    assert.equal(data[0], 0);
    assert.equal(data[2], 0);
    assert.equal(data[3], 255);
    assert.equal(data[5], 255);
  });

  test('Mat.cvtColor BGR 转灰度', async () => {
    const mat = cv.Mat.zeros(4, 4, cv.CV_8UC3);
    const gray = await mat.cvtColor(cv.ColorCode.Bgr2Gray);
    assert.equal(gray.channels, 1);
    assert.equal(gray.rows, 4);
    assert.equal(gray.cols, 4);
  });

  test('Mat.resize 缩放图像', async () => {
    const mat = cv.Mat.zeros(100, 100, cv.CV_8UC3);
    const resized = await mat.resize(50, 50);
    assert.equal(resized.rows, 50);
    assert.equal(resized.cols, 50);
  });

  test('Mat.resize 使用比例因子', async () => {
    const mat = cv.Mat.zeros(100, 100, cv.CV_8UC3);
    const resized = await mat.resize(0, 0, 0.5, 0.5);
    assert.equal(resized.rows, 50);
    assert.equal(resized.cols, 50);
  });

  test('Mat.gaussianBlur 高斯模糊', async () => {
    // 中心一个亮点，模糊后中心应减弱、周围应扩散
    const buf = Buffer.alloc(9 * 9, 0);
    buf[4 * 9 + 4] = 255; // 中心像素
    const mat = cv.Mat.fromBuffer(9, 9, cv.CV_8UC1, buf);
    const blurred = await mat.gaussianBlur(5, 5, 1.5);
    assert.equal(blurred.rows, 9);
    assert.equal(blurred.cols, 9);
    // 模糊后中心值应低于原来的 255
    assert.ok(blurred.data[4 * 9 + 4] < 255, '中心应被模糊减弱');
    // 模糊后周围原来的 0 应变为非零（能量扩散）
    assert.ok(blurred.data[3 * 9 + 4] > 0, '邻近像素应获得扩散能量');
  });

  test('Mat.canny 边缘检测', async () => {
    // 上半部分全白、下半部分全黑 → 水平边界上应有边缘
    const buf = Buffer.alloc(20 * 20, 0);
    for (let r = 0; r < 10; r++)
      for (let c = 0; c < 20; c++) buf[r * 20 + c] = 255;
    const mat = cv.Mat.fromBuffer(20, 20, cv.CV_8UC1, buf);
    const edges = await mat.canny(50, 150);
    assert.equal(edges.rows, 20);
    assert.equal(edges.cols, 20);
    assert.equal(edges.channels, 1);
    // 边界行 (row=9 或 row=10) 应有边缘像素 > 0
    let hasEdge = false;
    for (let c = 0; c < 20; c++) {
      if (edges.data[9 * 20 + c] > 0 || edges.data[10 * 20 + c] > 0) { hasEdge = true; break; }
    }
    assert.ok(hasEdge, '白黑边界处应检测到边缘');
  });

  test('Mat.flip 翻转图像', async () => {
    const buf = Buffer.from([1, 2, 3, 4]);
    const mat = cv.Mat.fromBuffer(2, 2, cv.CV_8UC1, buf);
    const flipped = await mat.flip(cv.FlipCode.Horizontal);
    const data = flipped.data;
    assert.equal(data[0], 2);
    assert.equal(data[1], 1);
    assert.equal(data[2], 4);
    assert.equal(data[3], 3);
  });

  test('Mat.crop 裁剪图像', async () => {
    const mat = cv.Mat.zeros(100, 100, cv.CV_8UC3);
    const cropped = await mat.crop(10, 10, 50, 50);
    assert.equal(cropped.rows, 50);
    assert.equal(cropped.cols, 50);
  });

  test('Mat.minMaxLoc 找最大最小值', async () => {
    const buf = Buffer.from([0, 50, 200, 100, 30, 255]);
    const mat = cv.Mat.fromBuffer(1, 6, cv.CV_8UC1, buf);
    const result = await mat.minMaxLoc();
    assert.equal(result.minVal, 0);
    assert.equal(result.maxVal, 255);
    assert.equal(result.minLoc.x, 0);
    assert.equal(result.maxLoc.x, 5);
  });

  test('Mat.inRange 颜色范围过滤', async () => {
    // [0,50,100,150,200,250] → 范围 [80,0,0,0]-[180,255,255,255] 内的像素应变 255，其余为 0
    const buf = Buffer.from([0, 50, 100, 150, 200, 250]);
    const mat = cv.Mat.fromBuffer(1, 6, cv.CV_8UC1, buf);
    const mask = await mat.inRange({v0: 80, v1: 0, v2: 0, v3: 0}, {v0: 180, v1: 0, v2: 0, v3: 0});
    assert.equal(mask.rows, 1);
    assert.equal(mask.cols, 6);
    assert.equal(mask.channels, 1);
    assert.equal(mask.data[0], 0,   '0   在范围外，应为 0');
    assert.equal(mask.data[1], 0,   '50  在范围外，应为 0');
    assert.equal(mask.data[2], 255, '100 在范围内，应为 255');
    assert.equal(mask.data[3], 255, '150 在范围内，应为 255');
    assert.equal(mask.data[4], 0,   '200 在范围外，应为 0');
    assert.equal(mask.data[5], 0,   '250 在范围外，应为 0');
  });

  test('Mat.bitwiseNot 按位取反', async () => {
    const buf = Buffer.from([0, 255, 128]);
    const mat = cv.Mat.fromBuffer(1, 3, cv.CV_8UC1, buf);
    const result = await mat.bitwiseNot();
    const data = result.data;
    assert.equal(data[0], 255);
    assert.equal(data[1], 0);
    assert.equal(data[2], 127);
  });

  test('Mat.bitwiseAnd 按位与', async () => {
    const buf1 = Buffer.from([0b11110000, 0b10101010]);
    const buf2 = Buffer.from([0b11001100, 0b01010101]);
    const mat1 = cv.Mat.fromBuffer(1, 2, cv.CV_8UC1, buf1);
    const mat2 = cv.Mat.fromBuffer(1, 2, cv.CV_8UC1, buf2);
    const result = await mat1.bitwiseAnd(mat2);
    const data = result.data;
    assert.equal(data[0], 0b11000000);
    assert.equal(data[1], 0b00000000);
  });

  test('Mat.bitwiseOr 按位或', async () => {
    const buf1 = Buffer.from([0b11000000]);
    const buf2 = Buffer.from([0b00000011]);
    const mat1 = cv.Mat.fromBuffer(1, 1, cv.CV_8UC1, buf1);
    const mat2 = cv.Mat.fromBuffer(1, 1, cv.CV_8UC1, buf2);
    const result = await mat1.bitwiseOr(mat2);
    assert.equal(result.data[0], 0b11000011);
  });

  test('Mat.split 分离通道', async () => {
    const mat = cv.Mat.zeros(5, 5, cv.CV_8UC3);
    const channels = await mat.split();
    assert.equal(channels.length, 3);
    for (const ch of channels) {
      assert.equal(ch.channels, 1);
      assert.equal(ch.rows, 5);
      assert.equal(ch.cols, 5);
    }
  });

  test('Mat.normalize 归一化', async () => {
    const buf = Buffer.from([0, 50, 100, 150, 200, 255]);
    const mat = cv.Mat.fromBuffer(1, 6, cv.CV_8UC1, buf);
    const normalized = await mat.normalize(0, 255, cv.NormType.MinMax);
    assert.equal(normalized.rows, 1);
    assert.equal(normalized.cols, 6);
    // MinMax 归一化：原来最小值→0，最大值→255
    assert.equal(normalized.data[0], 0,   '原始最小值 0 应映射到 0');
    assert.equal(normalized.data[5], 255, '原始最大值 255 应映射到 255');
    // 中间值应按比例缩放：100/255*255 ≈ 100
    assert.ok(normalized.data[2] > 80 && normalized.data[2] < 120, `中间值应约 100，实际 ${normalized.data[2]}`);
  });

  test('Mat.matchTemplate 模板匹配', async () => {
    const src = cv.Mat.zeros(100, 100, cv.CV_8UC1);
    const tmpl = cv.Mat.zeros(10, 10, cv.CV_8UC1);
    const result = await src.matchTemplate(tmpl, cv.TemplateMatchMode.Sqdiff);
    assert.equal(result.rows, 91);
    assert.equal(result.cols, 91);
  });

  test('Mat.morphologyEx 形态学操作', async () => {
    // 开运算 (Open = 先腐蚀后膨胀)：可去除小噪点
    // 在黑色背景上放一个大白色方块 + 一个小噪点
    const buf = Buffer.alloc(20 * 20, 0);
    for (let r = 5; r < 15; r++) for (let c = 5; c < 15; c++) buf[r * 20 + c] = 255; // 大块
    buf[1 * 20 + 1] = 255; // 小噪点
    const mat = cv.Mat.fromBuffer(20, 20, cv.CV_8UC1, buf);
    const kernel = cv.getStructuringElement(cv.MorphShape.Rect, 3, 3);
    const result = await mat.morphologyEx(cv.MorphType.Open, kernel);
    assert.equal(result.rows, 20);
    assert.equal(result.cols, 20);
    // 大块内部应保留（中心 (10,10) 应为 255）
    assert.equal(result.data[10 * 20 + 10], 255, '大块中心应保留');
    // 小噪点 (1,1) 应被开运算消除
    assert.equal(result.data[1 * 20 + 1], 0, '单像素噪点应被开运算消除');
  });
});
