'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const cv = require('../index.js');

describe('DNN 模块', () => {
  test('nmsBoxes 是自由函数', () => {
    assert.equal(typeof cv.nmsBoxes, 'function');
  });

  test('nmsBoxes 非极大值抑制', async () => {
    const bboxes = [
      { x: 0, y: 0, width: 100, height: 100 },
      { x: 5, y: 5, width: 100, height: 100 },
      { x: 200, y: 200, width: 100, height: 100 },
    ];
    const scores = [0.9, 0.8, 0.95];
    const result = await cv.nmsBoxes(bboxes, scores, 0.5, 0.5);
    assert.ok(Array.isArray(result));
    // 前两个高度重叠，应合并为一个；第三个独立
    assert.equal(result.length, 2);
  });

  test('nmsBoxes 高 NMS 阈值保留更多框', async () => {
    const bboxes = [
      { x: 0, y: 0, width: 100, height: 100 },
      { x: 200, y: 200, width: 100, height: 100 },
      { x: 400, y: 400, width: 100, height: 100 },
    ];
    const scores = [0.9, 0.8, 0.7];
    const result = await cv.nmsBoxes(bboxes, scores, 0.5, 0.9);
    assert.equal(result.length, 3);
  });

  test('nmsBoxes 过滤低分数框', async () => {
    const bboxes = [
      { x: 0, y: 0, width: 100, height: 100 },
      { x: 200, y: 200, width: 100, height: 100 },
    ];
    const scores = [0.9, 0.3]; // 第二个低于阈值
    const result = await cv.nmsBoxes(bboxes, scores, 0.5, 0.5);
    assert.equal(result.length, 1);
  });
});
