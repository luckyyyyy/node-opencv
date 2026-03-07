'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const cv = require('../index.js');

// 构造一个简单的 3x3 灰度 Mat
function gray3x3(values) {
  const buf = Buffer.from(values);
  return cv.Mat.fromBuffer(3, 3, cv.CV_8UC1, buf);
}

describe('Mat 新增操作', () => {
  test('medianBlur 中值滤波', async () => {
    const mat = gray3x3([0,0,0, 0,255,0, 0,0,0]);
    const out = await mat.medianBlur(3);
    assert.equal(out.rows, 3);
    assert.equal(out.cols, 3);
    // 中值滤波后中心噪点应被抑制
    assert.equal(out.data[4], 0);
  });

  test('addWeighted 图像融合', async () => {
    const a = gray3x3([100,100,100, 100,100,100, 100,100,100]);
    const b = gray3x3([50,50,50, 50,50,50, 50,50,50]);
    // 0.5*a + 0.5*b + 0 = 75
    const out = await a.addWeighted(0.5, b, 0.5, 0.0);
    assert.equal(out.rows, 3);
    assert.equal(out.data[0], 75);
  });

  test('equalizeHist 直方图均衡化', async () => {
    // 输入：0-99的线性斜坡，均衡化后应拉伸至全色阈
    const buf = Buffer.alloc(100);
    for (let i = 0; i < 100; i++) buf[i] = i;
    const mat = cv.Mat.fromBuffer(10, 10, cv.CV_8UC1, buf);
    const out = await mat.equalizeHist();
    assert.equal(out.rows, 10);
    assert.equal(out.cols, 10);
    // 均衡化后应显著拉伸：min 应接近 0，max 应上到 200+
    const d = out.data;
    const min = Math.min(...d);
    const max = Math.max(...d);
    assert.ok(min < 10, `min=${min} 应 < 10`);
    assert.ok(max > 200, `max=${max} 应 > 200`);
  });

  test('copyMakeBorder 添加边框', async () => {
    const mat = gray3x3([1,2,3, 4,5,6, 7,8,9]);
    const out = await mat.copyMakeBorder(1, 1, 1, 1, cv.BorderType.Constant, [0]);
    assert.equal(out.rows, 5);
    assert.equal(out.cols, 5);
    // 边框象素应为 0 (常量边界)
    assert.equal(out.data[0], 0, '左上角边框应为 0');
    assert.equal(out.data[4], 0, '右上角边框应为 0');
    // 原始内容应保留：(1,1)处应为原 mat[0,0]=1
    assert.equal(out.data[1 * 5 + 1], 1, '内容左上角应为 1');
    assert.equal(out.data[1 * 5 + 3], 3, '内容右上角应为 3');
    assert.equal(out.data[3 * 5 + 3], 9, '内容右下角应为 9');
  });

  test('absDiff 绝对差值', async () => {
    const a = gray3x3([100,100,100, 100,100,100, 100,100,100]);
    const b = gray3x3([60,60,60, 60,60,60, 60,60,60]);
    const out = await a.absDiff(b);
    assert.equal(out.data[0], 40);
  });

  test('add 矩阵加法', async () => {
    const a = gray3x3([10,20,30, 40,50,60, 70,80,90]);
    const b = gray3x3([1,2,3, 4,5,6, 7,8,9]);
    const out = await a.add(b);
    assert.equal(out.data[0], 11);
    assert.equal(out.data[4], 55);
  });

  test('subtract 矩阵减法', async () => {
    const a = gray3x3([100,100,100, 100,100,100, 100,100,100]);
    const b = gray3x3([10,10,10, 10,10,10, 10,10,10]);
    const out = await a.subtract(b);
    assert.equal(out.data[0], 90);
  });

  test('multiply 矩阵乘法(元素)', async () => {
    const a = gray3x3([2,2,2, 2,2,2, 2,2,2]);
    const b = gray3x3([3,3,3, 3,3,3, 3,3,3]);
    const out = await a.multiply(b);
    assert.equal(out.data[0], 6);
  });

  test('sobel 边缘检测', async () => {
    // 在 row=3-6, col=3-6 处有一个白色块
    const buf = Buffer.alloc(10 * 10, 0);
    for (let r = 3; r < 7; r++) for (let c = 3; c < 7; c++) buf[r * 10 + c] = 200;
    const mat = cv.Mat.fromBuffer(10, 10, cv.CV_8UC1, buf);
    const out = await mat.sobel(cv.CV_16S, 1, 0);
    assert.equal(out.rows, 10);
    assert.equal(out.cols, 10);
    // row=4 (white 区中间), col=3 (左边界 0→2000的过渡) 应有大的正梯度
    const shorts = new Int16Array(out.data.buffer, out.data.byteOffset, 10 * 10);
    const edgeVal = shorts[4 * 10 + 3];
    assert.ok(edgeVal > 100, `左边界梯度=${edgeVal} 应 > 100`);
    // col=7 (右边界 200→0的过渡) 应有大的负梯度
    const edgeValR = shorts[4 * 10 + 7];
    assert.ok(edgeValR < -100, `右边界梯度=${edgeValR} 应 < -100`);
  });

  test('laplacian 拉普拉斯算子', async () => {
    // 平坦图像的拉普拉斯应为全零
    const buf = Buffer.alloc(10 * 10, 128);
    const mat = cv.Mat.fromBuffer(10, 10, cv.CV_8UC1, buf);
    const out = await mat.laplacian(cv.CV_16S);
    assert.equal(out.rows, 10);
    assert.equal(out.cols, 10);
    const shorts = new Int16Array(out.data.buffer, out.data.byteOffset, 10 * 10);
    for (const v of shorts) assert.equal(v, 0, `平坦图像的拉普拉斯应为 0，实际得到 ${v}`);
  });

  test('bilateralFilter 双边滤波', async () => {
    const buf = Buffer.alloc(20 * 20, 128);
    const mat = cv.Mat.fromBuffer(20, 20, cv.CV_8UC1, buf);
    const out = await mat.bilateralFilter(5, 75, 75);
    assert.equal(out.rows, 20);
    assert.equal(out.cols, 20);
    // 均匀输入应保持不变
    for (const v of out.data) assert.equal(v, 128);
  });

  test('warpPerspective 透视变换', async () => {
    // 单位透视矩阵（恒等变换）应保持所有像素值
    const identityData = Buffer.from(new Float64Array([1,0,0, 0,1,0, 0,0,1]).buffer);
    const M = cv.Mat.fromBuffer(3, 3, cv.CV_64F, identityData);
    const buf = Buffer.alloc(10 * 10 * 3, 128);
    // 写入可识别的标志像素
    buf[0] = 42; buf[1] = 43; buf[2] = 44;
    buf[9 * 10 * 3 + 9 * 3] = 77; buf[9 * 10 * 3 + 9 * 3 + 1] = 88; buf[9 * 10 * 3 + 9 * 3 + 2] = 99;
    const src = cv.Mat.fromBuffer(10, 10, cv.CV_8UC3, buf);
    const out = await src.warpPerspective(M, 10, 10);
    assert.equal(out.rows, 10);
    assert.equal(out.cols, 10);
    // 恒等变换应保持像素值不变
    assert.equal(out.data[0], 42);
    assert.equal(out.data[1], 43);
    assert.equal(out.data[2], 44);
  });

  test('hconcat 水平拼接', async () => {
    const a = gray3x3([1,2,3, 4,5,6, 7,8,9]);
    const b = gray3x3([10,11,12, 13,14,15, 16,17,18]);
    const out = await cv.hconcat([a, b]);
    assert.equal(out.rows, 3);
    assert.equal(out.cols, 6);
    // 前 3 列来自 a，后 3 列来自 b
    assert.equal(out.data[0 * 6 + 0], 1,  'row0 col0 应为 a[0,0]=1');
    assert.equal(out.data[0 * 6 + 2], 3,  'row0 col2 应为 a[0,2]=3');
    assert.equal(out.data[0 * 6 + 3], 10, 'row0 col3 应为 b[0,0]=10');
    assert.equal(out.data[2 * 6 + 5], 18, 'row2 col5 应为 b[2,2]=18');
  });

  test('vconcat 垂直拼接', async () => {
    const a = gray3x3([1,2,3, 4,5,6, 7,8,9]);
    const b = gray3x3([10,11,12, 13,14,15, 16,17,18]);
    const out = await cv.vconcat([a, b]);
    assert.equal(out.rows, 6);
    assert.equal(out.cols, 3);
    // 前 3 行来自 a，后 3 行来自 b
    assert.equal(out.data[0 * 3 + 0], 1,  'row0 col0 应为 a[0,0]=1');
    assert.equal(out.data[2 * 3 + 2], 9,  'row2 col2 应为 a[2,2]=9');
    assert.equal(out.data[3 * 3 + 0], 10, 'row3 col0 应为 b[0,0]=10');
    assert.equal(out.data[5 * 3 + 2], 18, 'row5 col2 应为 b[2,2]=18');
  });

  test('getRotationMatrix2D 旋转矩阵', async () => {
    const M = cv.getRotationMatrix2D(50, 50, 45, 1.0);
    assert.equal(M.rows, 2);
    assert.equal(M.cols, 3);
    // 45° 旋转：cos(45°) = sin(45°) ≈ 0.7071
    const f64 = new Float64Array(M.data.buffer, M.data.byteOffset, 6);
    const cos45 = Math.cos(Math.PI / 4);
    assert.ok(Math.abs(f64[0] - cos45) < 1e-5, `M[0,0]=${f64[0]} 应≈${cos45}`);
    assert.ok(Math.abs(f64[1] - cos45) < 1e-5, `M[0,1]=${f64[1]} 应≈${cos45}`);
  });

  test('filter2D 自定义卷积', async () => {
    const buf = Buffer.alloc(10 * 10, 100);
    const src = cv.Mat.fromBuffer(10, 10, cv.CV_8UC1, buf);
    // 3x3 均值核 (1/9 * 1)
    const kernelData = Buffer.from(new Float32Array(9).fill(1 / 9).buffer);
    const kernel = cv.Mat.fromBuffer(3, 3, cv.CV_32F, kernelData);
    const out = await src.filter2D(-1, kernel);
    assert.equal(out.rows, 10);
    assert.equal(out.cols, 10);
    // 均値输入用均値核卷积，内部像素应保持 100
    assert.equal(out.data[5 * 10 + 5], 100);
  });

  test('merge 通道合并', async () => {
    const b = gray3x3([10,10,10, 10,10,10, 10,10,10]);
    const g = gray3x3([20,20,20, 20,20,20, 20,20,20]);
    const r = gray3x3([30,30,30, 30,30,30, 30,30,30]);
    const bgr = await cv.merge([b, g, r]);
    assert.equal(bgr.rows, 3);
    assert.equal(bgr.cols, 3);
    assert.equal(bgr.channels, 3);
    // BGR 布局: 第 0、3、... 字节为 B=10，第 1、4... 为 G=20，第 2、5... 为 R=30
    assert.equal(bgr.data[0], 10, 'pixel[0] B 应为 10');
    assert.equal(bgr.data[1], 20, 'pixel[0] G 应为 20');
    assert.equal(bgr.data[2], 30, 'pixel[0] R 应为 30');
  });
});
