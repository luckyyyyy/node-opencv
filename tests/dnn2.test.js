'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const cv = require('../index.js');

describe('DNN blobFromImage', () => {
  test('blobFromImage 从 RGB Mat 生成 blob', async () => {
    // 10x10 3-channel image
    const buf = Buffer.alloc(10 * 10 * 3, 128);
    const img = cv.Mat.fromBuffer(10, 10, cv.CV_8UC3, buf);
    const blob = await cv.blobFromImage(img, 1.0 / 255.0, 10, 10);
    // blob shape: [1, C, H, W] = 4D, total = 1*3*10*10 = 300
    assert.equal(blob.empty, false);
    assert.equal(blob.total, 300);
  });

  test('blobFromImage 指定均值', async () => {
    const buf = Buffer.alloc(8 * 8 * 3, 200);
    const img = cv.Mat.fromBuffer(8, 8, cv.CV_8UC3, buf);
    // scale=1.0, mean=[104,117,123], 输入值=200
    // B通道: 200-104=96, G通道: 200-117=83, R通道: 200-123=77
    const blob = await cv.blobFromImage(img, 1.0, 8, 8, [104, 117, 123]);
    assert.equal(blob.empty, false);
    assert.equal(blob.total, 192); // 1*3*8*8 = 192 元素
    const floats = new Float32Array(blob.data.buffer, blob.data.byteOffset, 192);
    // blob 布局: [B通道64个, G通道64个, R通道64个]
    assert.ok(Math.abs(floats[0]   -  96) < 1, `B通道应为96，实际${floats[0]}`);
    assert.ok(Math.abs(floats[64]  -  83) < 1, `G通道应为83，实际${floats[64]}`);
    assert.ok(Math.abs(floats[128] -  77) < 1, `R通道应为77，实际${floats[128]}`);
  });

  test('DNN_BACKEND_OPENCV 常量正确', () => {
    assert.equal(cv.DnnBackend.OpenCv, 3);
  });

  test('DNN_TARGET_CPU 常量正确', () => {
    assert.equal(cv.DnnTarget.Cpu, 0);
  });

  test('DNN_BACKEND_DEFAULT 常量正确', () => {
    assert.equal(cv.DnnBackend.Default, 0);
  });

  test('DNN_TARGET_CUDA 常量存在', () => {
    assert.ok(typeof cv.DnnTarget.Cuda === 'number');
  });
});
