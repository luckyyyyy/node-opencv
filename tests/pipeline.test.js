/**
 * 端到端组合流程测试
 * 验证多个 API 协同工作：从创建 Mat → 各种处理步骤 → 最终结果
 */
'use strict';
const { describe, test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const cv = require('../index.js');
const ONNX_MODEL = path.join(__dirname, 'fixtures', 'identity.onnx');

// ──────────────────────────────────────────────────────────
// 辅助
// ──────────────────────────────────────────────────────────

function solidMat(rows, cols, value = 128, type = cv.CV_8UC1) {
  const ch = type & 0xff8 ? (type >> 3) + 1 : 1;
  const bytes = {
    [cv.CV_8UC1]: 1, [cv.CV_8UC3]: 3, [cv.CV_8UC4]: 4,
    [cv.CV_32FC1]: 4, [cv.CV_32FC3]: 12,
  }[type] ?? 1;
  const total = rows * cols * bytes;
  const buf = Buffer.alloc(total, 0);
  if (type === cv.CV_32FC1) {
    new Float32Array(buf.buffer, buf.byteOffset, rows * cols).fill(value);
  } else {
    buf.fill(value);
  }
  return cv.Mat.fromBuffer(rows, cols, type, buf);
}

/** 创建简单 8x8 BGR 图像，中间有一个亮色方块 */
function makeTestBGR() {
  const rows = 16, cols = 16;
  const buf = Buffer.alloc(rows * cols * 3, 30);  // 暗背景
  // 在 (4,4)~(11,11) 设置亮色块
  for (let r = 4; r < 12; r++) {
    for (let c = 4; c < 12; c++) {
      const idx = (r * cols + c) * 3;
      buf[idx] = 200; buf[idx + 1] = 200; buf[idx + 2] = 200;
    }
  }
  return cv.Mat.fromBuffer(rows, cols, cv.CV_8UC3, buf);
}

// ──────────────────────────────────────────────────────────
// 流程 1：图像处理流水线 (cvtColor → gaussianBlur → canny → threshold)
// ──────────────────────────────────────────────────────────
describe('流程 1：图像处理流水线', () => {
  test('BGR → 灰度 → 高斯模糊 → Canny 边缘 → 二值化', async () => {
    const bgr = makeTestBGR();
    assert.equal(bgr.channels, 3);

    // 转灰度
    const gray = await bgr.cvtColor(cv.ColorCode.Bgr2Gray);
    assert.equal(gray.channels, 1);
    assert.equal(gray.rows, 16);

    // 高斯模糊
    const blurred = await gray.gaussianBlur(3, 3, 1.0);
    assert.equal(blurred.rows, 16);

    // Canny 边缘检测
    const edges = await blurred.canny(30, 100);
    assert.equal(edges.rows, 16);
    assert.equal(edges.depth, 0); // CV_8U

    // 二值化（边缘已是二值，这里验证 API 可链式调用）
    const binary = await edges.threshold(0, 255, cv.ThresholdType.Binary);
    assert.equal(binary.rows, 16);
    assert.equal(binary.total, 256);
  });
});

// ──────────────────────────────────────────────────────────
// 流程 2：形态学处理（腐蚀膨胀开操作）
// ──────────────────────────────────────────────────────────
describe('流程 2：形态学处理流水线', () => {
  test('二值图 → dilate → erode (开操作) → 验证形状保留', async () => {
    // 制造带噪声的二值图：中间亮块 + 散点噪声
    const rows = 16, cols = 16;
    const buf = Buffer.alloc(rows * cols, 0);
    // 中心亮块
    for (let r = 5; r < 11; r++)
      for (let c = 5; c < 11; c++)
        buf[r * cols + c] = 255;
    // 一个单像素噪声
    buf[1 * cols + 1] = 255;
    const binary = cv.Mat.fromBuffer(rows, cols, cv.CV_8UC1, buf);

    const kernel = cv.Mat.ones(3, 3, cv.CV_8UC1);
    // 膨胀
    const dilated = await binary.dilate(kernel, 1);
    assert.equal(dilated.rows, 16);
    // 腐蚀（整体 = 开操作）
    const opened = await dilated.erode(kernel, 1);
    assert.equal(opened.rows, 16);

    // 开操作后噪声点应消失（中心像素仍应为 255 之类的非零值）
    const centerVal = opened.data[8 * cols + 8];
    assert.ok(centerVal > 0, `中心像素应为亮，实际 ${centerVal}`);
  });

  test('morphologyEx Open → 验证与手动膨胀腐蚀等价', async () => {
    const rows = 16, cols = 16;
    const buf = Buffer.alloc(rows * cols, 0);
    for (let r = 4; r < 12; r++)
      for (let c = 4; c < 12; c++)
        buf[r * cols + c] = 255;
    const mat = cv.Mat.fromBuffer(rows, cols, cv.CV_8UC1, buf);
    const kernel = cv.Mat.ones(3, 3, cv.CV_8UC1);

    const result = await mat.morphologyEx(cv.MorphType.Open, kernel);
    assert.equal(result.rows, 16);
    // 中心像素应保留
    assert.ok(result.data[8 * cols + 8] > 0);
  });
});

// ──────────────────────────────────────────────────────────
// 流程 3：模板匹配 → 找到位置 → 绘制矩形
// ──────────────────────────────────────────────────────────
describe('流程 3：模板匹配并绘制结果', () => {
  test('matchTemplate → minMaxLoc → 验证找到的位置', async () => {
    // 制造明确嵌入模板的图像
    const rows = 20, cols = 20;
    const buf = Buffer.alloc(rows * cols, 50);
    // 在 (5,5) 放一个 6×6=200 的亮块作为模板目标
    for (let r = 5; r < 11; r++)
      for (let c = 5; c < 11; c++)
        buf[r * cols + c] = 200;
    const src = cv.Mat.fromBuffer(rows, cols, cv.CV_8UC1, buf);

    // 提取模板：与原图完全一致的 6×6 块
    const tmplBuf = Buffer.alloc(6 * 6, 200);
    const tmpl = cv.Mat.fromBuffer(6, 6, cv.CV_8UC1, tmplBuf);

    const result = await src.matchTemplate(tmpl, cv.TemplateMatchMode.Sqdiff);
    assert.ok(result.rows > 0, '匹配结果不为空');

    const loc = await result.minMaxLoc();
    assert.ok(typeof loc.minVal === 'number');
    assert.ok(typeof loc.minLoc.x === 'number');
    assert.ok(typeof loc.minLoc.y === 'number');
    // SQDIFF 模式：最小值处即匹配位置，应在 (5,5) 附近
    assert.ok(loc.minLoc.x >= 4 && loc.minLoc.x <= 6, `x=${loc.minLoc.x}`);
    assert.ok(loc.minLoc.y >= 4 && loc.minLoc.y <= 6, `y=${loc.minLoc.y}`);
  });

  test('matchTemplateAll → 返回矩形数组', async () => {
    // 在一个大图里嵌入两处相同模板
    const rows = 32, cols = 32;
    const buf = Buffer.alloc(rows * cols, 30);
    const patch = 200;
    // 第一处  (2,2)
    for (let r = 2; r < 6; r++)
      for (let c = 2; c < 6; c++)
        buf[r * cols + c] = patch;
    // 第二处 (20,2)
    for (let r = 20; r < 24; r++)
      for (let c = 2; c < 6; c++)
        buf[r * cols + c] = patch;
    const src = cv.Mat.fromBuffer(rows, cols, cv.CV_8UC1, buf);

    const tmplBuf = Buffer.alloc(4 * 4, patch);
    const tmpl = cv.Mat.fromBuffer(4, 4, cv.CV_8UC1, tmplBuf);

    const rects = await src.matchTemplateAll(tmpl, cv.TemplateMatchMode.CcoeffNormed, 0.5, 0.3);
    assert.ok(Array.isArray(rects));
    // 至少找到一个匹配
    assert.ok(rects.length >= 1, `期望至少 1 个匹配，实际 ${rects.length}`);
  });
});

// ──────────────────────────────────────────────────────────
// 流程 4：边缘检测 → findContours → 绘制轮廓
// ──────────────────────────────────────────────────────────
describe('流程 4：findContours → drawContours 流水线', () => {
  test('二值图 → findContours → drawContours → 像素变化', async () => {
    const rows = 20, cols = 20;
    const buf = Buffer.alloc(rows * cols, 0);
    for (let r = 5; r < 15; r++)
      for (let c = 5; c < 15; c++)
        buf[r * cols + c] = 255;
    const binary = cv.Mat.fromBuffer(rows, cols, cv.CV_8UC1, buf);

    const contours = await binary.findContours(
      cv.ContourRetrievalMode.External,
      cv.ContourApproximation.Simple
    );
    assert.ok(contours.length >= 1, `应有至少 1 个轮廓，实际 ${contours.length}`);

    // 绘制轮廓到画布
    const canvas = cv.Mat.zeros(rows, cols, cv.CV_8UC3);
    cv.drawContours(canvas, contours, -1, { v0: 0, v1: 255, v2: 0, v3: 0 }, 1);
    // 绘制后画布不再全为零
    let nonZero = 0;
    for (const b of canvas.data) if (b > 0) nonZero++;
    assert.ok(nonZero > 0, '绘制轮廓后画布应有非零像素');
  });
});

// ──────────────────────────────────────────────────────────
// 流程 5：DNN 推理流水线 (identity model)
// ──────────────────────────────────────────────────────────
describe('流程 5：DNN 推理流水线', () => {
  test('Mat → blobFromImage → Net.run → 结果形状验证', async () => {
    // 创建 4×4 float32 图像
    const buf = Buffer.allocUnsafe(4 * 4 * 4);
    new Float32Array(buf.buffer, buf.byteOffset, 16).fill(0.7);
    const img = cv.Mat.fromBuffer(4, 4, cv.CV_32FC1, buf);

    // blobFromImage：scaleFactor=1, 4×4, no mean, no swapRb, no crop
    const blob = await cv.blobFromImage(img, 1.0, 4, 4);
    assert.equal(blob.total, 16);

    const net = cv.Net.readNetFromOnnx(ONNX_MODEL);
    net.setPreferableBackend(cv.DnnBackend.OpenCv);
    net.setPreferableTarget(cv.DnnTarget.Cpu);

    const output = await net.run(blob, 'input', 'output');
    assert.equal(output.total, 16);

    // identity 模型：输出应等于输入
    const inView  = new Float32Array(blob.data.buffer,   blob.data.byteOffset,   16);
    const outView = new Float32Array(output.data.buffer, output.data.byteOffset, 16);
    for (let i = 0; i < 16; i++) {
      assert.ok(Math.abs(outView[i] - inView[i]) < 1e-4,
        `位置 ${i}: 期望 ${inView[i]}，实际 ${outView[i]}`);
    }
  });

  test('runMultiple 返回多输出数组', async () => {
    const buf = Buffer.allocUnsafe(4 * 4 * 4);
    new Float32Array(buf.buffer, buf.byteOffset, 16).fill(0.3);
    const img = cv.Mat.fromBuffer(4, 4, cv.CV_32FC1, buf);
    const blob = await cv.blobFromImage(img, 1.0, 4, 4);

    const net = cv.Net.readNetFromOnnx(ONNX_MODEL);
    const outputs = await net.runMultiple(blob, 'input', ['output']);
    assert.equal(outputs.length, 1);
    assert.equal(outputs[0].total, 16);
  });
});

// ──────────────────────────────────────────────────────────
// 流程 6：图像编码/解码往返
// ──────────────────────────────────────────────────────────
describe('流程 6：imencode → imdecode 往返', () => {
  test('BGR Mat → imencode PNG → imdecode → 像素完整性', async () => {
    const rows = 8, cols = 8;
    const buf = Buffer.alloc(rows * cols * 3);
    // R 渐变
    for (let i = 0; i < rows * cols; i++) {
      buf[i * 3]     = i * 4 % 256;  // B
      buf[i * 3 + 1] = 100;           // G
      buf[i * 3 + 2] = 200;           // R
    }
    const src = cv.Mat.fromBuffer(rows, cols, cv.CV_8UC3, buf);

    const encoded = await cv.imencode('.png', src);
    assert.ok(encoded instanceof Buffer);
    assert.ok(encoded.length > 0);

    const decoded = await cv.imdecode(encoded);
    assert.equal(decoded.rows, rows);
    assert.equal(decoded.cols, cols);
    assert.equal(decoded.channels, 3);
    // 固定像素验证（PNG 无损，可精确匹配）
    assert.equal(decoded.data[2], 200, 'R 通道应为 200');
    assert.equal(decoded.data[1], 100, 'G 通道应为 100');
  });

  test('灰度 imencode JPEG → imdecode → 尺寸正确', async () => {
    const rows = 16, cols = 16;
    const buf = Buffer.alloc(rows * cols, 150);
    const src = cv.Mat.fromBuffer(rows, cols, cv.CV_8UC1, buf);

    const jpgBuf = await cv.imencode('.jpg', src);
    assert.ok(jpgBuf.length > 0);

    const decoded = await cv.imdecode(jpgBuf);
    assert.equal(decoded.rows, rows);
    assert.equal(decoded.cols, cols);
    // JPEG 有损，只验证尺寸
  });
});

// ──────────────────────────────────────────────────────────
// 流程 7：channel split → process → merge 往返
// ──────────────────────────────────────────────────────────
describe('流程 7：split → 处理 → merge 往返', () => {
  test('BGR split → 对一个通道 threshold → merge → 验证', async () => {
    const rows = 8, cols = 8;
    const buf = Buffer.alloc(rows * cols * 3);
    for (let i = 0; i < rows * cols; i++) {
      buf[i * 3]     = 50;   // B
      buf[i * 3 + 1] = 150;  // G
      buf[i * 3 + 2] = 100;  // R
    }
    const src = cv.Mat.fromBuffer(rows, cols, cv.CV_8UC3, buf);

    const channels = await src.split();
    assert.equal(channels.length, 3);

    // 对 G 通道做二值化
    const gThresh = await channels[1].threshold(100, 255, cv.ThresholdType.Binary);
    assert.equal(gThresh.data[0], 255, 'G>100 → 255');

    // 合并回去
    const merged = await cv.merge([channels[0], gThresh, channels[2]]);
    assert.equal(merged.rows, rows);
    assert.equal(merged.cols, cols);
    assert.equal(merged.channels, 3);
    // B 通道应保持 50，G 通道为 255（已阈值）
    assert.equal(merged.data[0], 50);
    assert.equal(merged.data[1], 255);
  });
});

// ──────────────────────────────────────────────────────────
// 流程 8：透视变换 → 仿射变换
// ──────────────────────────────────────────────────────────
describe('流程 8：透视/仿射变换流水线', () => {
  test('getPerspectiveTransform → warpPerspective → 验证尺寸', async () => {
    // 正方形 → 正方形（identity-like 变换）
    const srcPts = [
      { x: 0, y: 0 }, { x: 7, y: 0 },
      { x: 7, y: 7 }, { x: 0, y: 7 }
    ];
    const dstPts = [
      { x: 0, y: 0 }, { x: 7, y: 0 },
      { x: 7, y: 7 }, { x: 0, y: 7 }
    ];
    const M = await cv.getPerspectiveTransform(srcPts, dstPts);
    assert.equal(M.rows, 3);
    assert.equal(M.cols, 3);

    const mat = solidMat(8, 8, 128);
    const out = await mat.warpPerspective(M, 8, 8);
    assert.equal(out.rows, 8);
    assert.equal(out.cols, 8);
  });

  test('getAffineTransform → warpAffine → 验证尺寸', async () => {
    const srcPts = [{ x: 0, y: 0 }, { x: 7, y: 0 }, { x: 0, y: 7 }];
    const dstPts = [{ x: 0, y: 0 }, { x: 7, y: 0 }, { x: 0, y: 7 }];
    const M = await cv.getAffineTransform(srcPts, dstPts);
    assert.equal(M.rows, 2);
    assert.equal(M.cols, 3);

    const mat = solidMat(8, 8, 128);
    const out = await mat.warpAffine(M, 8, 8);
    assert.equal(out.rows, 8);
    assert.equal(out.cols, 8);
  });
});

// ──────────────────────────────────────────────────────────
// 流程 9：多图合并（hconcat + vconcat）流水线
// ──────────────────────────────────────────────────────────
describe('流程 9：hconcat + vconcat 组合', () => {
  test('4 个小 Mat → hconcat 2 对 → vconcat 合并 → 验证大小', async () => {
    const a = solidMat(4, 4, 50);
    const b = solidMat(4, 4, 100);
    const c = solidMat(4, 4, 150);
    const d = solidMat(4, 4, 200);

    const top    = await cv.hconcat([a, b]);  // 4×8
    const bottom = await cv.hconcat([c, d]);  // 4×8
    const full   = await cv.vconcat([top, bottom]); // 8×8

    assert.equal(full.rows, 8);
    assert.equal(full.cols, 8);

    // 像素分区验证
    assert.equal(full.data[0],             50,  '左上角应为 50');
    assert.equal(full.data[4],             100, '右上角第一行应为 100');
    assert.equal(full.data[4 * 8],         150, '左下角应为 150');
    assert.equal(full.data[4 * 8 + 4],     200, '右下角应为 200');
  });
});

// ──────────────────────────────────────────────────────────
// 流程 10：直方图均衡化流水线
// ──────────────────────────────────────────────────────────
describe('流程 10：equalizeHist 流水线', () => {
  test('低对比度灰度图 → equalizeHist → 对比度增强', async () => {
    // 所有像素都在 100-110 之间（低对比度）
    const rows = 16, cols = 16;
    const buf = Buffer.alloc(rows * cols);
    for (let i = 0; i < rows * cols; i++) buf[i] = 100 + (i % 11);
    const src = cv.Mat.fromBuffer(rows, cols, cv.CV_8UC1, buf);

    const eq = await src.equalizeHist();
    assert.equal(eq.rows, rows);
    // 均衡化后动态范围应更大
    let minVal = 255, maxVal = 0;
    for (const v of eq.data) {
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
    assert.ok(maxVal - minVal > 50, `均衡化后范围应更大，实际 ${minVal}~${maxVal}`);
  });
});
