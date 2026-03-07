'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const cv = require('../index.js');

const ONNX_MODEL = path.join(__dirname, 'fixtures', 'identity.onnx');

// 构建一个 4x4 float32 图像，blobFromImage 得到 1x1x4x4 blob
async function makeBlob(value = 0.5) {
  const data = new Float32Array(16).fill(value);
  const img = cv.Mat.fromBuffer(4, 4, cv.CV_32FC1, Buffer.from(data.buffer));
  return cv.blobFromImage(img, 1.0, 4, 4);
}

describe('Net 类', () => {
  test('Net.readNetFromOnnx 加载模型', () => {
    const net = cv.Net.readNetFromOnnx(ONNX_MODEL);
    assert.ok(net, 'Net 实例应存在');
  });

  test('Net.readNetFromOnnx 不存在文件应抛出错误', () => {
    assert.throws(
      () => cv.Net.readNetFromOnnx('/nonexistent/model.onnx'),
      (e) => e instanceof Error,
    );
  });

  test('Net.getUnconnectedOutLayersNames 返回输出层名称', () => {
    const net = cv.Net.readNetFromOnnx(ONNX_MODEL);
    const names = net.getUnconnectedOutLayersNames();
    assert.ok(Array.isArray(names));
    assert.ok(names.length >= 1);
    assert.equal(names[0], 'output');
  });

  test('Net.setPreferableBackend 不抛出错误', () => {
    const net = cv.Net.readNetFromOnnx(ONNX_MODEL);
    // OpenCv backend 在所有平台都支持
    assert.doesNotThrow(() => net.setPreferableBackend(cv.DnnBackend.OpenCv));
  });

  test('Net.setPreferableTarget CPU 不抛出错误', () => {
    const net = cv.Net.readNetFromOnnx(ONNX_MODEL);
    assert.doesNotThrow(() => net.setPreferableTarget(cv.DnnTarget.Cpu));
  });

  test('Net.setPreferableBackend Default 不抛出错误', () => {
    const net = cv.Net.readNetFromOnnx(ONNX_MODEL);
    assert.doesNotThrow(() => net.setPreferableBackend(cv.DnnBackend.Default));
  });

  test('Net.run 推理返回输出 Mat', async () => {
    const net = cv.Net.readNetFromOnnx(ONNX_MODEL);
    net.setPreferableBackend(cv.DnnBackend.OpenCv);
    net.setPreferableTarget(cv.DnnTarget.Cpu);
    const blob = await makeBlob(0.5);
    const out = await net.run(blob, 'input', 'output');
    assert.ok(!out.empty, '输出 Mat 不应为空');
    assert.equal(out.total, 16, '输出应有 16 个元素(1x1x4x4)');
    // Identity 输出应与输入相同
    const floats = new Float32Array(out.data.buffer, out.data.byteOffset, 16);
    for (const v of floats) {
      assert.ok(Math.abs(v - 0.5) < 1e-4, `Identity 输出应等于输入 0.5，实际 ${v}`);
    }
  });

  test('Net.run 可省略 outputName 参数', async () => {
    const net = cv.Net.readNetFromOnnx(ONNX_MODEL);
    const blob = await makeBlob(1.0);
    // outputName undefined → 自动选第一个输出层
    const out = await net.run(blob, 'input', null);
    assert.ok(!out.empty);
  });

  test('Net.runMultiple 返回多输出数组', async () => {
    const net = cv.Net.readNetFromOnnx(ONNX_MODEL);
    const blob = await makeBlob(0.25);
    const outs = await net.runMultiple(blob, 'input', ['output']);
    assert.ok(Array.isArray(outs));
    assert.equal(outs.length, 1);
    assert.ok(!outs[0].empty);
    assert.equal(outs[0].total, 16);
    const floats = new Float32Array(outs[0].data.buffer, outs[0].data.byteOffset, 16);
    for (const v of floats) {
      assert.ok(Math.abs(v - 0.25) < 1e-4, `Identity 输出应等于输入 0.25，实际 ${v}`);
    }
  });

  test('Net.run 配合 blobFromImage swapRb=true', async () => {
    // 3通道图像，swapRb 交换 R/B 通道
    const buf = Buffer.alloc(4 * 4 * 3, 0);
    // B=100, G=150, R=200
    for (let i = 0; i < 16; i++) {
      buf[i * 3 + 0] = 100;
      buf[i * 3 + 1] = 150;
      buf[i * 3 + 2] = 200;
    }
    const img = cv.Mat.fromBuffer(4, 4, cv.CV_8UC3, buf);
    const blob = await cv.blobFromImage(img, 1.0 / 255.0, 4, 4, null, true); // swapRb=true
    assert.ok(!blob.empty);
    assert.equal(blob.total, 48, '3*4*4=48 元素');
    // swapRb=true 时，blob 中 R 通道在前，B 通道在后
    const floats = new Float32Array(blob.data.buffer, blob.data.byteOffset, 48);
    // 第 0 个通道（R=200): 200/255 ≈ 0.784
    assert.ok(Math.abs(floats[0] - 200 / 255) < 1e-3, `R通道应≈${200/255}，实际${floats[0]}`);
    // 第 2 个通道（B=100): 100/255 ≈ 0.392
    assert.ok(Math.abs(floats[32] - 100 / 255) < 1e-3, `B通道应≈${100/255}，实际${floats[32]}`);
  });

  test('Net.run 配合 blobFromImage crop=true', async () => {
    // crop=true 时裁剪中心区域后缩放到目标尺寸
    const buf = Buffer.alloc(8 * 8 * 3, 128);
    const img = cv.Mat.fromBuffer(8, 8, cv.CV_8UC3, buf);
    const blob = await cv.blobFromImage(img, 1.0 / 255.0, 4, 4, null, false, true);
    assert.ok(!blob.empty);
    assert.equal(blob.total, 48, 'crop 后 3*4*4=48 元素');
  });
});
