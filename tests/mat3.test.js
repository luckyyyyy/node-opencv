'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const cv = require('../index.js');
const path = require('node:path');

const FIXTURES = path.join(__dirname, 'fixtures');
const BUILDING_JPG = path.join(FIXTURES, 'building.jpg');

// Helper: 3x3 single-channel Mat
function gray(rows, cols, fill) {
  const buf = Buffer.alloc(rows * cols, fill ?? 128);
  return cv.Mat.fromBuffer(rows, cols, cv.CV_8UC1, buf);
}

describe('Mat 补充覆盖测试', () => {
  test('Mat.eye 恒等矩阵', () => {
    const m = cv.Mat.eye(3, 3, cv.CV_32F);
    assert.equal(m.rows, 3);
    assert.equal(m.cols, 3);
    // diagonal = 1, off-diagonal = 0 (float32 buffer)
    const data = new Float32Array(m.data.buffer, m.data.byteOffset, 9);
    assert.ok(Math.abs(data[0] - 1) < 1e-6);  // [0,0]
    assert.ok(Math.abs(data[4] - 1) < 1e-6);  // [1,1]
    assert.ok(Math.abs(data[8] - 1) < 1e-6);  // [2,2]
    assert.ok(Math.abs(data[1]) < 1e-6);       // [0,1] = 0
  });

  test('Mat.copyTo 同步复制', () => {
    const src = gray(4, 4, 200);
    const dst = new cv.Mat();
    src.copyTo(dst);
    assert.equal(dst.rows, 4);
    assert.equal(dst.cols, 4);
    assert.equal(dst.data[0], 200);
  });

  test('Mat.reshape 改变形状', () => {
    const buf = Buffer.alloc(12, 0);
    for (let i = 0; i < 12; i++) buf[i] = i;
    const mat = cv.Mat.fromBuffer(3, 4, cv.CV_8UC1, buf);
    const reshaped = mat.reshape(1, 2); // 2 rows, auto cols = 6
    assert.equal(reshaped.rows, 2);
    assert.equal(reshaped.cols, 6);
  });

  test('Mat.dilate 膨胀', async () => {
    const kernel = cv.getStructuringElement(cv.MorphShape.Rect, 3, 3);
    // 中心一个白色像素，膨胀后应扭展为 3x3 块
    const buf = Buffer.alloc(7 * 7, 0);
    buf[3 * 7 + 3] = 255;
    const mat = cv.Mat.fromBuffer(7, 7, cv.CV_8UC1, buf);
    const out = await mat.dilate(kernel);
    assert.equal(out.rows, 7);
    assert.equal(out.cols, 7);
    // 中心及相邻 3x3 区域应全为 255
    assert.equal(out.data[3 * 7 + 3], 255, '中心应为 255');
    assert.equal(out.data[2 * 7 + 2], 255, '左上邻居应为 255');
    assert.equal(out.data[4 * 7 + 4], 255, '右下邻居应为 255');
    // 角落 (0,0) 距中心太远，应持续为 0
    assert.equal(out.data[0], 0, '角落应为 0');
  });

  test('Mat.erode 腐蚀', async () => {
    const kernel = cv.getStructuringElement(cv.MorphShape.Rect, 3, 3);
    // 7x7 全白图像，腐蚀后边缘应变为 0
    const buf = Buffer.alloc(7 * 7, 255);
    const mat = cv.Mat.fromBuffer(7, 7, cv.CV_8UC1, buf);
    const out = await mat.erode(kernel);
    assert.equal(out.rows, 7);
    assert.equal(out.cols, 7);
    // 中心远离边界，3x3 kernel 能完全拟合，应保持 255
    assert.equal(out.data[3 * 7 + 3], 255, '中心应保持 255');
    // 角落 (0,0) kernel 无法完全拟合，应变为 0
    assert.equal(out.data[0], 0, '角落腐蚀应为 0');
  });

  test('Mat.warpAffine 仿射变换', async () => {
    // 2x3 单位仿射矩阵（恒等变换）
    const affineData = Buffer.from(new Float64Array([1, 0, 0, 0, 1, 0]).buffer);
    const M = cv.Mat.fromBuffer(2, 3, cv.CV_64F, affineData);
    const buf = Buffer.alloc(20 * 20 * 3, 100);
    // 写入标志像素用于验证展放保留
    buf[0] = 77; buf[1] = 88; buf[2] = 99;
    buf[(10 * 20 + 10) * 3] = 11; buf[(10 * 20 + 10) * 3 + 1] = 22; buf[(10 * 20 + 10) * 3 + 2] = 33;
    const src = cv.Mat.fromBuffer(20, 20, cv.CV_8UC3, buf);
    const out = await src.warpAffine(M, 20, 20);
    assert.equal(out.rows, 20);
    assert.equal(out.cols, 20);
    // 恒等变换应展放保留所有像素
    assert.equal(out.data[0], 77);
    assert.equal(out.data[1], 88);
    assert.equal(out.data[2], 99);
    assert.equal(out.data[(10 * 20 + 10) * 3], 11);
    assert.equal(out.data[(10 * 20 + 10) * 3 + 1], 22);
    assert.equal(out.data[(10 * 20 + 10) * 3 + 2], 33);
  });

  test('Mat.matchTemplateAll 真实图像模板匹配', async () => {
    // 从 building.jpg 裁剪一块作为模板，匹配结果应在原位置找到精确匹配
    const src = await cv.imread(BUILDING_JPG);
    // 裁剪 (x=100, y=100, w=100, h=100) 的区域作为模板
    const tpl = await src.crop(100, 100, 100, 100);
    const results = await src.matchTemplateAll(tpl, cv.TemplateMatchMode.CcoeffNormed, 0.95, 0.1);
    assert.ok(Array.isArray(results));
    assert.equal(results.length, 1, `应找到正好 1 个匹配，实际 ${results.length}`);
    const m = results[0];
    assert.ok(Math.abs(m.x - 100) < 3, `x 应接近 100，实际 ${m.x}`);
    assert.ok(Math.abs(m.y - 100) < 3, `y 应接近 100，实际 ${m.y}`);
    assert.equal(m.width, 100);
    assert.equal(m.height, 100);
  });

  test('Mat.transpose 转置', async () => {
    // 一个 [1,2,3; 4,5,6] 的 2x3 矩阵转置应变为 [1,4; 2,5; 3,6] 的 3x2
    const buf = Buffer.alloc(2 * 3, 0);
    for (let i = 0; i < 6; i++) buf[i] = i + 1;
    const mat = cv.Mat.fromBuffer(2, 3, cv.CV_8UC1, buf);
    const out = await mat.transpose();
    assert.equal(out.rows, 3);
    assert.equal(out.cols, 2);
    // 转置后: [0][0]=1, [0][1]=4, [1][0]=2, [1][1]=5, [2][0]=3, [2][1]=6
    assert.equal(out.data[0 * 2 + 0], 1, 'T[0,0] 应为 1');
    assert.equal(out.data[0 * 2 + 1], 4, 'T[0,1] 应为 4');
    assert.equal(out.data[1 * 2 + 0], 2, 'T[1,0] 应为 2');
    assert.equal(out.data[1 * 2 + 1], 5, 'T[1,1] 应为 5');
    assert.equal(out.data[2 * 2 + 0], 3, 'T[2,0] 应为 3');
    assert.equal(out.data[2 * 2 + 1], 6, 'T[2,1] 应为 6');
  });

  test('countNonZero 非零元素计数', async () => {
    const buf = Buffer.from([0, 0, 1, 0, 2, 0, 0, 3, 0]);
    const mat = cv.Mat.fromBuffer(3, 3, cv.CV_8UC1, buf);
    assert.equal(await cv.countNonZero(mat), 3);
  });

  test('mean 均值计算', async () => {
    const buf = Buffer.alloc(4, 100);
    const mat = cv.Mat.fromBuffer(2, 2, cv.CV_8UC1, buf);
    const m = await cv.mean(mat);
    assert.ok(Array.isArray(m));
    assert.equal(m.length, 4);
    assert.ok(Math.abs(m[0] - 100) < 1);
  });
});
