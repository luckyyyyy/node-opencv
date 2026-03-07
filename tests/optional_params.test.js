/**
 * 可选参数分支覆盖测试
 * 确保每个可选参数路径至少被调用一次（有实参 vs 无实参）
 */
'use strict';
const { describe, test } = require('node:test');
const assert = require('node:assert/strict');
const cv = require('../index.js');

const m = (r = 8, c = 8, v = 128) => cv.Mat.fromBuffer(r, c, cv.CV_8UC1, Buffer.alloc(r * c, v));
const k = () => cv.Mat.ones(3, 3, cv.CV_8UC1);

// ──────────────────────────────────────────────────────────
// 表格驱动：只检查 out.rows/cols 的简单可选参数路径
// [名称, async () => number, expected]
// ──────────────────────────────────────────────────────────
const rowCases = [
  // gaussianBlur — sigmaY
  ['gaussianBlur 无 sigmaY',                 async () => (await m().gaussianBlur(3,3,1.0)).rows,          8],
  ['gaussianBlur 指定 sigmaY',               async () => (await m().gaussianBlur(3,3,1.0,2.0)).rows,      8],
  // canny — apertureSize 和 l2Gradient
  ['canny 无可选参数',                        async () => (await m().canny(50,150)).rows,                  8],
  ['canny apertureSize=5',                   async () => (await m().canny(50,150,5)).rows,                8],
  ['canny l2Gradient=true',                  async () => (await m().canny(50,150,3,true)).rows,           8],
  ['canny l2Gradient=false',                 async () => (await m().canny(50,150,3,false)).rows,          8],
  // dilate / erode — iterations
  ['dilate 无 iterations',                   async () => (await m().dilate(k())).rows,                    8],
  ['dilate iterations=3',                    async () => (await m().dilate(k(),3)).rows,                  8],
  ['erode 无 iterations',                    async () => (await m(8,8,200).erode(k())).rows,              8],
  ['erode iterations=3',                     async () => (await m(8,8,200).erode(k(),3)).rows,            8],
  // morphologyEx — iterations
  ['morphologyEx 无 iterations',             async () => (await m(8,8,200).morphologyEx(cv.MorphType.Open,k())).rows,   8],
  ['morphologyEx iterations=2',              async () => (await m(8,8,200).morphologyEx(cv.MorphType.Open,k(),2)).rows, 8],
  // sobel — 可选参数 ksize/scale/delta
  ['sobel 无可选参数',                        async () => (await m().sobel(cv.CV_32FC1,1,0)).rows,         8],
  ['sobel 指定 ksize=3',                     async () => (await m().sobel(cv.CV_32FC1,1,0,3)).rows,       8],
  ['sobel 指定 scale=2 delta=10',            async () => (await m().sobel(cv.CV_32FC1,1,0,3,2.0,10.0)).rows, 8],
  // laplacian — 可选参数 ksize/scale/delta
  ['laplacian 无可选参数',                    async () => (await m().laplacian(cv.CV_32FC1)).rows,         8],
  ['laplacian 指定 ksize=3',                 async () => (await m().laplacian(cv.CV_32FC1,3)).rows,       8],
  ['laplacian 指定 scale=1.5 delta=5',       async () => (await m().laplacian(cv.CV_32FC1,3,1.5,5.0)).rows, 8],
  // resize — interpolation 可选参数
  ['resize 无 interpolation (放大)',          async () => (await m().resize(16,16)).rows,                 16],
  ['resize InterpolationFlag.Nearest',       async () => (await m().resize(4,4,null,null,cv.InterpolationFlag.Nearest)).rows, 4],
  ['resize InterpolationFlag.Cubic',         async () => (await m().resize(4,4,null,null,cv.InterpolationFlag.Cubic)).rows,   4],
  ['resize 使用 fx/fy 缩放因子',              async () => (await m().resize(0,0,2.0,2.0)).rows,            16],
];

describe('简单可选参数路径覆盖', () => {
  for (const [name, run, expected] of rowCases) {
    test(name, async () => assert.equal(await run(), expected));
  }
});

// ──────────────────────────────────────────────────────────
// convertTo — alpha 和 beta（检查实际像素值）
// ──────────────────────────────────────────────────────────
describe('convertTo 可选参数', () => {
  test('convertTo(rtype) — 仅类型', async () => {
    const out = await m(2, 2, 100).convertTo(cv.CV_32FC1);
    assert.equal(out.depth, cv.CV_32FC1 & 7);
    assert.equal(out.rows, 2);
  });

  test('convertTo(rtype, alpha) — 指定 alpha', async () => {
    const out = await m(1, 4, 10).convertTo(cv.CV_32FC1, 2.0);
    const view = new Float32Array(out.data.buffer, out.data.byteOffset, 4);
    assert.ok(Math.abs(view[0] - 20.0) < 1e-4, `期望 20.0，实际 ${view[0]}`);
  });

  test('convertTo(rtype, alpha, beta) — alpha + beta 偏移', async () => {
    const out = await m(1, 4, 10).convertTo(cv.CV_32FC1, 2.0, 5.0); // 10*2+5=25
    const view = new Float32Array(out.data.buffer, out.data.byteOffset, 4);
    assert.ok(Math.abs(view[0] - 25.0) < 1e-4, `期望 25.0，实际 ${view[0]}`);
  });
});

// ──────────────────────────────────────────────────────────
// warpAffine — flags 和 borderMode
// ──────────────────────────────────────────────────────────
describe('warpAffine 可选参数', () => {
  function makeM() {
    const buf = Buffer.allocUnsafe(6 * 4);
    const f = new Float32Array(buf.buffer, buf.byteOffset, 6);
    f.set([1, 0, 0, 0, 1, 0]);
    return cv.Mat.fromBuffer(2, 3, cv.CV_32FC1, buf);
  }

  test('warpAffine 无可选参数',                               async () => assert.equal((await m().warpAffine(makeM(), 8, 8)).rows, 8));
  test('warpAffine 指定 flags=InterpolationFlag.Nearest',      async () => assert.equal((await m().warpAffine(makeM(), 8, 8, cv.InterpolationFlag.Nearest)).rows, 8));
  test('warpAffine 指定 borderMode=BorderType.Replicate',      async () => assert.equal((await m().warpAffine(makeM(), 8, 8, cv.InterpolationFlag.Linear, cv.BorderType.Replicate)).rows, 8));

  test('warpAffine WARP_FILL_OUTLIERS 常量已导出', () => {
    assert.equal(typeof cv.WARP_FILL_OUTLIERS, 'number');
    assert.equal(typeof cv.WARP_INVERSE_MAP,   'number');
  });
});

// ──────────────────────────────────────────────────────────
// warpPerspective — flags 和 borderMode
// ──────────────────────────────────────────────────────────
describe('warpPerspective 可选参数', () => {
  function makeM() {
    const buf = Buffer.allocUnsafe(9 * 8);
    const d = new Float64Array(buf.buffer, buf.byteOffset, 9);
    d.set([1, 0, 0,  0, 1, 0,  0, 0, 1]);
    return cv.Mat.fromBuffer(3, 3, cv.CV_64FC1, buf);
  }

  test('warpPerspective 无可选参数',                          async () => assert.equal((await m().warpPerspective(makeM(), 8, 8)).rows, 8));
  test('warpPerspective 指定 flags=InterpolationFlag.Cubic',   async () => assert.equal((await m().warpPerspective(makeM(), 8, 8, cv.InterpolationFlag.Cubic)).rows, 8));
  test('warpPerspective borderMode=BorderType.Reflect',       async () => assert.equal((await m().warpPerspective(makeM(), 8, 8, cv.InterpolationFlag.Linear, cv.BorderType.Reflect)).rows, 8));
});

// ──────────────────────────────────────────────────────────
// bitwiseAnd/Or/Not — mask
// ──────────────────────────────────────────────────────────
describe('bitwise mask 可选参数', () => {
  const mask = () => cv.Mat.fromBuffer(4, 4, cv.CV_8UC1, Buffer.alloc(16, 255));

  test('bitwiseAnd 带 mask', async () => {
    assert.equal((await cv.Mat.fromBuffer(4,4,cv.CV_8UC1,Buffer.alloc(16,0xFF)).bitwiseAnd(cv.Mat.fromBuffer(4,4,cv.CV_8UC1,Buffer.alloc(16,0x0F)), mask())).data[0], 0x0F);
  });

  test('bitwiseOr 带 mask', async () => {
    assert.equal((await cv.Mat.fromBuffer(4,4,cv.CV_8UC1,Buffer.alloc(16,0xF0)).bitwiseOr(cv.Mat.fromBuffer(4,4,cv.CV_8UC1,Buffer.alloc(16,0x0F)), mask())).data[0], 0xFF);
  });

  test('bitwiseNot 带 mask', async () => {
    assert.equal((await cv.Mat.fromBuffer(4,4,cv.CV_8UC1,Buffer.alloc(16,0b10101010)).bitwiseNot(mask())).data[0], ~0b10101010 & 0xFF);
  });
});

// ──────────────────────────────────────────────────────────
// normalize — dtype 可选参数
// ──────────────────────────────────────────────────────────
describe('normalize dtype 可选参数', () => {
  test('normalize 无 dtype', async () => {
    assert.equal((await m(1, 4, 100).normalize(0, 255, cv.NormType.MinMax)).rows, 1);
  });

  test('normalize 指定 dtype=CV_32FC1', async () => {
    const out = await m(1, 4, 100).normalize(0.0, 1.0, cv.NormType.MinMax, cv.CV_32FC1);
    assert.equal(out.matType, cv.CV_32FC1);
    assert.equal(out.rows, 1);
  });
});

// ──────────────────────────────────────────────────────────
// addWeighted — dtype 可选参数
// ──────────────────────────────────────────────────────────
describe('addWeighted dtype 可选参数', () => {
  test('addWeighted 无 dtype', async () => {
    assert.equal((await m(4,4,100).addWeighted(0.5, m(4,4,50), 0.5, 0)).rows, 4);
  });

  test('addWeighted 指定 dtype=CV_32FC1', async () => {
    const fa = await m(4,4,100).convertTo(cv.CV_32FC1);
    const fb = await m(4,4,50).convertTo(cv.CV_32FC1);
    const out = await fa.addWeighted(0.5, fb, 0.5, 0, cv.CV_32FC1);
    assert.equal(out.matType, cv.CV_32FC1);
  });
});

// ──────────────────────────────────────────────────────────
// multiply — scale 可选参数
// ──────────────────────────────────────────────────────────
describe('multiply scale 可选参数', () => {
  test('multiply 无 scale', async () => {
    assert.equal((await m(4,4,2).multiply(m(4,4,3))).data[0], 6);
  });

  test('multiply 指定 scale=0.01', async () => {
    const bA = Buffer.allocUnsafe(4 * 4 * 2); new Int16Array(bA.buffer, bA.byteOffset, 16).fill(100);
    const bB = Buffer.allocUnsafe(4 * 4 * 2); new Int16Array(bB.buffer, bB.byteOffset, 16).fill(100);
    const out = await cv.Mat.fromBuffer(4,4,cv.CV_16SC1,bA).multiply(cv.Mat.fromBuffer(4,4,cv.CV_16SC1,bB), 0.01);
    assert.ok(new Int16Array(out.data.buffer, out.data.byteOffset, 16)[0] > 0);
  });
});

// ──────────────────────────────────────────────────────────
// copyMakeBorder — value 数组
// ──────────────────────────────────────────────────────────
describe('copyMakeBorder value 可选参数', () => {
  test('copyMakeBorder 无 value', async () => {
    const out = await m(4,4,128).copyMakeBorder(2,2,2,2,cv.BorderType.Constant);
    assert.equal(out.rows, 8); assert.equal(out.cols, 8);
  });

  test('copyMakeBorder 指定 value=[50]', async () => {
    const out = await m(4,4,128).copyMakeBorder(2,2,2,2,cv.BorderType.Constant,[50]);
    assert.equal(out.rows, 8);
    assert.equal(out.data[0], 50, `边框值应为 50，实际 ${out.data[0]}`);
  });
});

// ──────────────────────────────────────────────────────────
// filter2D — delta 可选参数
// ──────────────────────────────────────────────────────────
describe('filter2D delta 可选参数', () => {
  function makeKernel3x3() {
    const buf = Buffer.allocUnsafe(9 * 4);
    new Float32Array(buf.buffer, buf.byteOffset, 9).fill(1 / 9);
    return cv.Mat.fromBuffer(3, 3, cv.CV_32FC1, buf);
  }

  test('filter2D 无 delta', async () => {
    assert.equal((await m(8,8,120).filter2D(-1, makeKernel3x3())).rows, 8);
  });

  test('filter2D 指定 delta=10', async () => {
    const out = await m(8, 8, 120).filter2D(-1, makeKernel3x3(), 10.0);
    assert.equal(out.rows, 8);
    assert.ok(out.data[4] >= 128 && out.data[4] <= 132, `期望约 130，实际 ${out.data[4]}`);
  });
});

// ──────────────────────────────────────────────────────────
// threshold — 多种 ThresholdType 变体（表格驱动）
// ──────────────────────────────────────────────────────────
describe('threshold ThresholdType 变体', () => {
  const thCases = [
    ['Binary',      m(1,4,100), 50, 255, cv.ThresholdType.Binary,     255],
    ['BinaryInv',   m(1,4,100), 50, 255, cv.ThresholdType.BinaryInv,  0],
    ['Trunc',       m(1,4,200), 150, 255, cv.ThresholdType.Trunc,     150],
    ['ToZero',      m(1,4,200), 100, 255, cv.ThresholdType.ToZero,    200],
    ['ToZeroInv',   m(1,4,50),  100, 255, cv.ThresholdType.ToZeroInv, 50],
  ];
  for (const [name, mat, thresh, maxval, type, expected] of thCases) {
    test(`ThresholdType.${name}`, async () => assert.equal((await mat.threshold(thresh, maxval, type)).data[0], expected));
  }
});

// ──────────────────────────────────────────────────────────
// VideoCapture.set() — 属性设置
// ──────────────────────────────────────────────────────────
describe('VideoCapture.set() 属性设置', () => {
  test('set CAP_PROP_POS_FRAMES 到 0（对摄像头不适用但不崩溃）', () => {
    try {
      const cap = cv.VideoCapture.open('/dev/null');
      assert.equal(typeof cap.set(cv.CAP_PROP_POS_FRAMES, 0), 'boolean');
      cap.release();
    } catch {
      // 无法打开设备也是可接受的
    }
  });
});
