'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const os = require('node:os');
const path = require('node:path');
const fs = require('node:fs');
const cv = require('../index.js');

// CAP_PROP_* 常量测试 (无需真实视频文件)
describe('VideoCapture/Writer 常量', () => {
  test('CAP_PROP_FPS 值正确', () => {
    assert.equal(cv.CAP_PROP_FPS, 5);
  });
  test('CAP_PROP_FRAME_WIDTH 值正确', () => {
    assert.equal(cv.CAP_PROP_FRAME_WIDTH, 3);
  });
  test('CAP_PROP_FRAME_HEIGHT 值正确', () => {
    assert.equal(cv.CAP_PROP_FRAME_HEIGHT, 4);
  });
  test('CAP_PROP_FRAME_COUNT 值正确', () => {
    assert.equal(cv.CAP_PROP_FRAME_COUNT, 7);
  });
  test('CAP_PROP_POS_MSEC 值正确', () => {
    assert.equal(cv.CAP_PROP_POS_MSEC, 0);
  });
  test('CAP_PROP_POS_FRAMES 值正确', () => {
    assert.equal(cv.CAP_PROP_POS_FRAMES, 1);
  });
});

describe('VideoWriter + VideoCapture', () => {
  const tmpFile = path.join(os.tmpdir(), `cv_test_${Date.now()}.avi`);

  test('VideoWriter.open + write + release', async () => {
    const writer = cv.VideoWriter.open(tmpFile, 'MJPG', 10, 20, 20, true);
    assert.ok(writer.isOpened());

    // 写5帧 20x20 BGR 黑色帧
    const frameBuf = Buffer.alloc(20 * 20 * 3, 0);
    const frame = cv.Mat.fromBuffer(20, 20, cv.CV_8UC3, frameBuf);
    for (let i = 0; i < 5; i++) await writer.write(frame);
    writer.release();
    assert.ok(!writer.isOpened());

    // 文件应写出
    assert.ok(fs.existsSync(tmpFile));
  });

  test('VideoCapture.open + isOpened + get + read + release', async () => {
    // 文件必须从上一个测试创建
    const cap = cv.VideoCapture.open(tmpFile);
    assert.ok(cap.isOpened());

    // get() 返回数字
    const fps = cap.get(cv.CAP_PROP_FPS);
    assert.ok(typeof fps === 'number');
    assert.ok(fps > 0);

    const w = cap.get(cv.CAP_PROP_FRAME_WIDTH);
    const h = cap.get(cv.CAP_PROP_FRAME_HEIGHT);
    assert.ok(w > 0);
    assert.ok(h > 0);

    // read 至少第一帧
    const frame = await cap.read();
    assert.ok(frame !== null && frame !== undefined);
    assert.ok(frame.rows > 0);

    cap.release();
    assert.ok(!cap.isOpened());
  });

  test('VideoCapture 不存在文件应返回 isOpened=false 或抛出', () => {
    try {
      const cap = cv.VideoCapture.open('/nonexistent/path/video.mp4');
      assert.ok(!cap.isOpened());
      cap.release();
    } catch (e) {
      // 抛出也是合理行为
      assert.ok(e instanceof Error);
    }
  });

  // 清理临时文件
  test('cleanup', () => {
    if (fs.existsSync(tmpFile)) fs.unlinkSync(tmpFile);
    assert.ok(true);
  });
});
