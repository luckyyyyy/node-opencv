# node-opencv-rs

[![npm version](https://img.shields.io/npm/v/node-opencv-rs.svg)](https://www.npmjs.com/package/node-opencv-rs)
[![CI](https://github.com/luckyyyyy/node-opencv/actions/workflows/ci.yml/badge.svg)](https://github.com/luckyyyyy/node-opencv/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js >= 18](https://img.shields.io/badge/node-%3E%3D18-brightgreen.svg)](https://nodejs.org)

高性能 Node.js [OpenCV](https://opencv.org) 绑定，基于 Rust 和 [napi-rs](https://napi.rs) 实现。所有耗时操作均在工作线程中执行，不阻塞事件循环。

[English Documentation](./README.md)

---

## 目录

- [功能特性](#功能特性)
- [环境要求](#环境要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [API 参考](#api-参考)
- [从源码构建](#从源码构建)
- [贡献指南](#贡献指南)

---

## 功能特性

- **`Mat` 类** — 核心矩阵类型：创建、克隆、复制、类型转换、重塑，以及原始像素数据访问
- **图像 I/O** — `imread`、`imwrite`、`imencode`、`imdecode`（异步，不阻塞事件循环）
- **图像处理** — 缩放、颜色转换、模糊（高斯 / 中值 / 双边）、阈值（含自适应阈值）、形态学操作（膨胀、腐蚀、开运算、闭运算、梯度、顶帽、黑帽）、Canny 边缘检测、Sobel/Laplacian 导数、直方图均衡化、边界填充等
- **算术与逻辑** — `add`、`subtract`、`multiply`、`absDiff`、`addWeighted`、`bitwiseAnd`、`bitwiseOr`、`bitwiseNot`、通道 `split` / `merge`、`normalize`、`hconcat` / `vconcat`、`countNonZero`、`mean`
- **几何变换** — 仿射变换、透视变换、翻转、裁剪、旋转矩阵、仿射变换矩阵、单应矩阵计算
- **特征检测** — 轮廓查找、轮廓面积/周长、外接矩形、最小外接矩形、凸包、多边形逼近、矩特征、角点检测、霍夫直线/圆检测
- **模板匹配** — `matchTemplate`、内置 NMS 的 `matchTemplateAll`
- **绘图** — 画线、矩形、圆形、椭圆、轮廓、多边形填充、文字渲染
- **深度学习 (DNN)** — `Net` 类支持 ONNX、Caffe、Darknet 模型；`blobFromImage`、`nmsBoxes`
- **视频** — `VideoCapture`（文件与摄像头）和 `VideoWriter`
- **完整 TypeScript 类型** — 内置 `index.d.ts` 类型声明
- **可取消异步操作** — 所有异步方法均支持可选的 `AbortSignal` 取消信号

## 环境要求

| 依赖 | 版本要求 |
|---|---|
| Node.js | ≥ 18 |
| OpenCV | 4.x |
| libclang | 任意近期版本（仅构建时需要） |

> **Linux**: `sudo apt install libopencv-dev libclang-dev`
> **macOS**: `brew install opencv llvm`
> **Windows**: 安装 OpenCV 4 并配置 `OPENCV_INCLUDE_PATHS`、`OPENCV_LINK_LIBS`、`OPENCV_LINK_PATHS` 环境变量

## 安装

**Linux x64 (glibc)** 和 **Windows x64 (MSVC)** 平台提供预编译二进制文件，无需 Rust 工具链。

```bash
npm install node-opencv-rs
```

若目标平台无预编译版本，`npm install` 将自动尝试从源码编译——参见[从源码构建](#从源码构建)。

## 快速开始

```js
const cv = require('node-opencv-rs');

// 读取图像
const mat = await cv.imread('./photo.jpg');
console.log(mat.rows, mat.cols, mat.channels); // 例如: 1080 1920 3

// 转换为灰度图并保存
const gray = await mat.cvtColor(cv.ColorCode.Bgr2Gray);
await cv.imwrite('./gray.png', gray);

// 模板匹配
const template = await cv.imread('./template.png');
const matches = await mat.matchTemplateAll(template, cv.TemplateMatchMode.CcoeffNormed, 0.85, 0.1);
console.log(matches); // [{ x, y, width, height }, ...]

// 从 Buffer 读取 / 编码
const buf = await cv.imencode('.jpg', mat);
const decoded = await cv.imdecode(buf);

// DNN 推理（ONNX 格式）
const net = cv.Net.readNetFromOnnx('./model.onnx');
net.setPreferableBackend(cv.DnnBackend.OpenCv);
net.setPreferableTarget(cv.DnnTarget.Cpu);
const blob = await cv.blobFromImage(mat, 1 / 255.0, 640, 640);
const output = await net.run(blob);

// 视频读取
const cap = cv.VideoCapture.open('./video.mp4');
while (cap.isOpened()) {
  const frame = await cap.read();
  if (!frame) break;
  // 处理每一帧...
}
cap.release();
```

## API 参考

> 所有异步方法返回 `Promise<T>`，最后一个参数可传 `AbortSignal` 用于取消操作。

### `Mat` 类

| 方法 | 说明 |
|---|---|
| `new Mat()` | 创建空矩阵 |
| `Mat.zeros(rows, cols, type)` | 创建全零矩阵 |
| `Mat.ones(rows, cols, type)` | 创建全一矩阵 |
| `Mat.eye(rows, cols, type)` | 创建单位矩阵 |
| `Mat.fromBuffer(rows, cols, type, buf)` | 从原始 `Buffer` 创建矩阵 |
| `.rows` / `.cols` / `.channels` | 维度信息 |
| `.matType` / `.depth` / `.elemSize` | 类型信息 |
| `.empty` | 是否为空矩阵 |
| `.size` | `{ width, height }` |
| `.total` | 元素总数 |
| `.data` | 像素原始数据（`Buffer`） |
| `.clone()` | 深拷贝 |
| `.copyTo(dst)` | 复制到另一个 `Mat` |
| `.convertTo(rtype, alpha?, beta?)` | 转换元素类型 |
| `.reshape(cn, rows?)` | 不复制数据地重塑形状 |
| `.release()` | 显式释放内存 |
| **异步图像操作** | `cvtColor`、`resize`、`gaussianBlur`、`medianBlur`、`bilateralFilter`、`threshold`、`canny`、`dilate`、`erode`、`morphologyEx`、`warpAffine`、`warpPerspective`、`flip`、`crop`、`inRange`、`normalize`、`equalizeHist`、`copyMakeBorder`、`filter2D`、`sobel`、`laplacian` |
| **异步算术操作** | `add`、`subtract`、`multiply`、`absDiff`、`addWeighted`、`bitwiseAnd`、`bitwiseOr`、`bitwiseNot`、`split`、`merge` |
| **异步分析操作** | `matchTemplate`、`matchTemplateAll`、`minMaxLoc` |

### 图像 I/O

```ts
imread(path: string, flags?: ImreadFlag): Promise<Mat>
imwrite(path: string, mat: Mat): Promise<boolean>
imencode(ext: string, mat: Mat): Promise<Buffer>   // 如 '.png'、'.jpg'
imdecode(buffer: Buffer, flags?: ImreadFlag): Promise<Mat>
```

### 特征检测

```ts
// 异步
findContours(mat, mode: ContourRetrievalMode, method: ContourApproximation): Promise<Point[][]>
moments(mat, binaryImage?: boolean): Promise<MomentsResult>
goodFeaturesToTrack(mat, maxCorners, qualityLevel, minDistance): Promise<PointF64[]>
houghLines(mat, rho, theta, threshold): Promise<HoughLine[]>
houghLinesP(mat, rho, theta, threshold, minLineLength?, maxLineGap?): Promise<LineSegment[]>
houghCircles(mat, dp, minDist, ...): Promise<Circle[]>

// 同步
contourArea(contour: Point[]): number
arcLength(contour: Point[], closed: boolean): number
boundingRect(contour: Point[]): Rect
minAreaRect(contour: Point[]): RotatedRect
convexHull(points: Point[]): Point[]
approxPolyDP(contour: Point[], epsilon: number, closed: boolean): Point[]
```

### 绘图

```ts
// 所有绘图函数均同步原地修改 mat，返回 void
drawLine(mat: Mat, pt1: Point, pt2: Point, color: Scalar, thickness?: number, lineType?: LineType): void
drawRectangle(mat: Mat, rect: Rect, color: Scalar, thickness?: number, lineType?: LineType): void
drawCircle(mat: Mat, center: Point, radius: number, color: Scalar, thickness?: number, lineType?: LineType): void
drawEllipse(mat: Mat, center: Point, axes: Size, angle: number, startAngle: number, endAngle: number, color: Scalar, thickness?: number, lineType?: LineType): void
drawContours(mat: Mat, contours: Point[][], contourIdx: number, color: Scalar, thickness?: number, lineType?: LineType): void
fillPoly(mat: Mat, pts: Point[][], color: Scalar, lineType?: LineType): void
putText(mat: Mat, text: string, org: Point, fontFace: HersheyFont, fontScale: number, color: Scalar, thickness?: number, lineType?: LineType): void
getTextSize(text: string, fontFace: HersheyFont, fontScale: number, thickness?: number): TextSize
```

### 深度学习 (DNN)

```ts
// 加载模型
Net.readNetFromOnnx(path: string): Net
Net.readNetFromCaffe(protoPath: string, modelPath: string): Net
Net.readNetFromDarknet(cfgPath: string, weightsPath: string): Net

// 配置
net.setPreferableBackend(backend: DnnBackend): void
net.setPreferableTarget(target: DnnTarget): void
net.getUnconnectedOutLayersNames(): string[]

// 推理（在单次加锁中原子性地执行 setInput + forward，避免竞态条件）
net.run(blob: Mat, inputName?: string, outputName?: string, abortSignal?: AbortSignal): Promise<Mat>
net.runMultiple(blob: Mat, inputName: string | null, outputNames: string[], abortSignal?: AbortSignal): Promise<Mat[]>

// 预处理
blobFromImage(image: Mat, scaleFactor?: number, width?: number, height?: number, mean?: Scalar, swapRb?: boolean, crop?: boolean, abortSignal?: AbortSignal): Promise<Mat>
nmsBoxes(bboxes: Rect[], scores: number[], scoreThreshold: number, nmsThreshold: number, abortSignal?: AbortSignal): Promise<number[]>
```

### 视频

```ts
// 读取视频
const cap = VideoCapture.open('./video.mp4');  // 或 VideoCapture.open(0) 打开摄像头
cap.isOpened(): boolean
cap.read(): Promise<Mat | null>          // 流结束时返回 null
cap.get(propId: number): number          // CAP_PROP_* 常量
cap.set(propId: number, value: number): boolean
cap.release(): void

// 写入视频
const writer = VideoWriter.open('out.mp4', 'mp4v', 30, 1920, 1080);
writer.write(frame: Mat): void
writer.release(): void
```

### 常用类型

```ts
interface Point     { x: number; y: number }
interface PointF64  { x: number; y: number }
interface Rect      { x: number; y: number; width: number; height: number }
interface Size      { width: number; height: number }
interface Scalar    { v0: number; v1: number; v2: number; v3: number }
interface RotatedRect { center: PointF64; size: SizeF64; angle: number }
```

### 常量

矩阵类型常量：`CV_8U`、`CV_8UC1`、`CV_8UC2`、`CV_8UC3`、`CV_8UC4`、`CV_16U`、`CV_16S`、`CV_32S`、`CV_32F`、`CV_64F` 等

视频捕获常量：`CAP_PROP_FPS`、`CAP_PROP_FRAME_WIDTH`、`CAP_PROP_FRAME_HEIGHT`、`CAP_PROP_FRAME_COUNT`、`CAP_PROP_POS_FRAMES`、`CAP_PROP_POS_MSEC` 等

枚举类型：`ColorCode`、`ImreadFlag`、`ThresholdType`、`InterpolationFlag`、`MorphType`、`MorphShape`、`BorderType`、`TemplateMatchMode`、`FlipCode`、`LineType`、`HersheyFont`、`NormType`、`DnnBackend`、`DnnTarget`、`ContourRetrievalMode`、`ContourApproximation`、`AdaptiveThresholdType`

## 从源码构建

**1. 安装系统依赖**

```bash
# Ubuntu / Debian
sudo apt-get install cmake libopencv-dev llvm clang libclang-dev

# macOS
brew install opencv llvm

# Windows — 安装 OpenCV 4，然后配置以下环境变量：
# OPENCV_INCLUDE_PATHS, OPENCV_LINK_LIBS, OPENCV_LINK_PATHS
```

**2. 安装 Rust**

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**3. 克隆并构建**

```bash
git clone https://github.com/luckyyyyy/node-opencv.git
cd node-opencv-rs
npm install
npm run build       # 发布构建（较慢，已优化）
# 或
npm run build:debug # 调试构建（较快，用于开发）
npm test
```

> Linux 可能需要设置 `LIBCLANG_PATH`：
> ```bash
> export LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu
> ```

## 贡献指南

欢迎提交 Issue 和 Pull Request！

- 所有异步操作必须使用 `napi::AsyncTask`，**禁止** 使用 `tokio::spawn`
- 避免不必要的 `.clone()` 调用
- 提交前请执行 `npm run build:debug && npm test`

---

## 许可证

[MIT](./LICENSE)
