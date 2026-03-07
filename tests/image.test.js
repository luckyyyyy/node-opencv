'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const cv = require('../index.js');
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');

const FIXTURES = path.join(__dirname, 'fixtures');
const LENA_JPG = path.join(FIXTURES, 'lena.jpg');

describe('imgcodecs 图像 IO', () => {
  test('imencode 编码为 PNG', async () => {
    const mat = cv.Mat.zeros(10, 10, cv.CV_8UC3);
    const buf = await cv.imencode('.png', mat);
    assert.ok(buf instanceof Buffer);
    assert.ok(buf.length > 0);
    // PNG 魔数: 89 50 4E 47
    assert.equal(buf[0], 0x89);
    assert.equal(buf[1], 0x50);
    assert.equal(buf[2], 0x4E);
    assert.equal(buf[3], 0x47);
  });

  test('imencode 编码为 JPEG', async () => {
    const mat = cv.Mat.zeros(10, 10, cv.CV_8UC3);
    const buf = await cv.imencode('.jpg', mat);
    assert.ok(buf instanceof Buffer);
    assert.ok(buf.length > 0);
    // JPEG 魔数: FF D8
    assert.equal(buf[0], 0xFF);
    assert.equal(buf[1], 0xD8);
  });

  test('imdecode 解码真实 JPEG', async () => {
    const buf = fs.readFileSync(LENA_JPG);
    const decoded = await cv.imdecode(buf);
    assert.equal(decoded.rows, 512);
    assert.equal(decoded.cols, 512);
    assert.equal(decoded.channels, 3);
    assert.ok(!decoded.empty);
  });

  test('imdecode 灰度解码真实 JPEG', async () => {
    const buf = fs.readFileSync(LENA_JPG);
    const decoded = await cv.imdecode(buf, cv.ImreadFlag.Grayscale);
    assert.equal(decoded.channels, 1);
    assert.equal(decoded.rows, 512);
    assert.equal(decoded.cols, 512);
  });

  test('imwrite 写入文件', async () => {
    const tmpFile = path.join(os.tmpdir(), `test_imwrite_${Date.now()}.png`);
    try {
      const mat = cv.Mat.zeros(10, 10, cv.CV_8UC3);
      const ok = await cv.imwrite(tmpFile, mat);
      assert.equal(ok, true);
      assert.ok(fs.existsSync(tmpFile));
      const stat = fs.statSync(tmpFile);
      assert.ok(stat.size > 0);
    } finally {
      if (fs.existsSync(tmpFile)) fs.unlinkSync(tmpFile);
    }
  });

  test('imread 读取真实 JPEG', async () => {
    const loaded = await cv.imread(LENA_JPG);
    assert.equal(loaded.rows, 512);
    assert.equal(loaded.cols, 512);
    assert.equal(loaded.channels, 3);
    assert.ok(!loaded.empty);
  });

  test('imread 读取真实 JPEG 灰度模式', async () => {
    const loaded = await cv.imread(LENA_JPG, cv.ImreadFlag.Grayscale);
    assert.equal(loaded.rows, 512);
    assert.equal(loaded.cols, 512);
    assert.equal(loaded.channels, 1);
  });

  test('imread 不存在文件抛出错误', async () => {
    await assert.rejects(
      cv.imread('/nonexistent/path/image.png'),
      /Failed to read image/
    );
  });
});

describe('工具函数', () => {
  test('getTickFrequency 返回正数', () => {
    const freq = cv.getTickFrequency();
    assert.equal(typeof freq, 'number');
    assert.ok(freq > 0);
  });

  test('getTickCount 返回数字', () => {
    const count = cv.getTickCount();
    assert.equal(typeof count, 'number');
    assert.ok(count > 0);
  });

  test('getBuildInformation 返回字符串', () => {
    const info = cv.getBuildInformation();
    assert.equal(typeof info, 'string');
    assert.ok(info.length > 0);
  });
});
