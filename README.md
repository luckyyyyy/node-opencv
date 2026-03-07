# node-opencv-rs

[![npm version](https://img.shields.io/npm/v/node-opencv-rs.svg)](https://www.npmjs.com/package/node-opencv-rs)
[![CI](https://github.com/luckyyyyy/node-opencv/actions/workflows/ci.yml/badge.svg)](https://github.com/luckyyyyy/node-opencv/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js >= 18](https://img.shields.io/badge/node-%3E%3D18-brightgreen.svg)](https://nodejs.org)

High-performance [OpenCV](https://opencv.org) bindings for Node.js, implemented in Rust via [napi-rs](https://napi.rs). All heavy operations run off the main thread using native async tasks — the event loop is never blocked.

[中文文档](./README.zh-CN.md)

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Building from Source](#building-from-source)
- [Contributing](#contributing)

---

## Features

- **`Mat` class** — Core matrix type: create, clone, copy, convert, reshape, and access raw pixel data
- **Image I/O** — `imread`, `imwrite`, `imencode`, `imdecode` (async, non-blocking)
- **Image processing** — resize, color conversion, blur (Gaussian / median / bilateral), threshold (including adaptive), morphology (dilate, erode, open, close, gradient, top-hat, black-hat), Canny edge detection, Sobel/Laplacian derivatives, histogram equalization, border padding, and more
- **Arithmetic & logic** — `add`, `subtract`, `multiply`, `absDiff`, `addWeighted`, `bitwiseAnd`, `bitwiseOr`, `bitwiseNot`, channel `split` / `merge`, `normalize`, `hconcat` / `vconcat`, `countNonZero`, `mean`
- **Geometric transforms** — `warpAffine`, `warpPerspective`, `flip`, `crop`, `getRotationMatrix2D`, `getAffineTransform`, `getPerspectiveTransform`
- **Feature detection** — `findContours`, `contourArea`, `arcLength`, `boundingRect`, `minAreaRect`, `convexHull`, `approxPolyDP`, `moments`, `goodFeaturesToTrack`, `houghLines`, `houghLinesP`, `houghCircles`
- **Template matching** — `matchTemplate`, `matchTemplateAll` with built-in NMS
- **Drawing** — `drawLine`, `drawRectangle`, `drawCircle`, `drawEllipse`, `drawContours`, `fillPoly`, `putText`
- **DNN** — `Net` class with support for ONNX, Caffe and Darknet models; `blobFromImage`, `nmsBoxes`
- **Video** — `VideoCapture` (file & camera) and `VideoWriter`
- **Fully typed** — Complete TypeScript declarations included (`index.d.ts`)
- **Non-blocking** — Every async method accepts an optional `AbortSignal` for cancellation

### Requirements

| Dependency | Version |
|---|---|
| Node.js | ≥ 18 |
| OpenCV | 4.x |
| libclang | any recent version (build-time only) |

> **Linux**: `sudo apt install libopencv-dev libclang-dev`
> **macOS**: `brew install opencv llvm`
> **Windows**: Install OpenCV 4 and set `OPENCV_INCLUDE_PATHS`, `OPENCV_LINK_LIBS`, `OPENCV_LINK_PATHS` environment variables.

### Installation

Prebuilt binaries are published for **Linux x64 (glibc)**, **macOS arm64 (Apple Silicon)**, and **Windows x64 (MSVC)**. No Rust toolchain required for these platforms.

```bash
npm install node-opencv-rs
```

If a prebuilt binary is not available for your platform, `npm install` will attempt to compile from source — see [Building from Source](#building-from-source).

### Quick Start

```js
const cv = require('node-opencv-rs');

// Read an image
const mat = await cv.imread('./photo.jpg');
console.log(mat.rows, mat.cols, mat.channels); // e.g. 1080 1920 3

// Convert to grayscale and write back
const gray = await mat.cvtColor(cv.ColorCode.Bgr2Gray);
await cv.imwrite('./gray.png', gray);

// Template matching
const template = await cv.imread('./template.png');
const matches = await mat.matchTemplateAll(template, cv.TemplateMatchMode.CcoeffNormed, 0.85, 0.1);
console.log(matches); // [{ x, y, width, height }, ...]

// DNN inference (ONNX)
const net = cv.Net.readNetFromOnnx('./model.onnx');
net.setPreferableBackend(cv.DnnBackend.OpenCv);
net.setPreferableTarget(cv.DnnTarget.Cpu);
const blob = await cv.blobFromImage(mat, 1 / 255.0, 640, 640);
const output = await net.run(blob);
```

### API Reference

> All async methods return `Promise<T>` and accept an optional `AbortSignal` as the last argument.

#### `Mat` class

| Method | Description |
|---|---|
| `new Mat()` | Create an empty matrix |
| `Mat.zeros(rows, cols, type)` | Create a zero-filled matrix |
| `Mat.ones(rows, cols, type)` | Create a matrix filled with ones |
| `Mat.eye(rows, cols, type)` | Create an identity matrix |
| `Mat.fromBuffer(rows, cols, type, buf)` | Create a matrix from a raw `Buffer` |
| `.rows` / `.cols` / `.channels` | Dimensions |
| `.matType` / `.depth` / `.elemSize` | Type info |
| `.empty` | Whether the matrix is empty |
| `.size` | `{ width, height }` |
| `.total` | Total number of elements |
| `.data` | Raw pixel data as `Buffer` |
| `.clone()` | Deep copy |
| `.copyTo(dst)` | Copy into another `Mat` |
| `.convertTo(rtype, alpha?, beta?)` | Convert element type |
| `.reshape(cn, rows?)` | Reshape without copying data |
| `.release()` | Explicitly free memory |
| **Async image ops** | `cvtColor`, `resize`, `gaussianBlur`, `medianBlur`, `bilateralFilter`, `threshold`, `canny`, `dilate`, `erode`, `morphologyEx`, `warpAffine`, `warpPerspective`, `flip`, `crop`, `inRange`, `normalize`, `equalizeHist`, `copyMakeBorder`, `filter2D`, `sobel`, `laplacian` |
| **Async arithmetic** | `add`, `subtract`, `multiply`, `absDiff`, `addWeighted`, `bitwiseAnd`, `bitwiseOr`, `bitwiseNot`, `split`, `merge` |
| **Async analysis** | `matchTemplate`, `matchTemplateAll`, `minMaxLoc` |

#### Image I/O

```ts
imread(path: string, flags?: ImreadFlag): Promise<Mat>
imwrite(path: string, mat: Mat): Promise<boolean>
imencode(ext: string, mat: Mat): Promise<Buffer>   // e.g. '.png', '.jpg'
imdecode(buffer: Buffer, flags?: ImreadFlag): Promise<Mat>
```

#### Feature Detection

```ts
// Async
findContours(mat, mode: ContourRetrievalMode, method: ContourApproximation): Promise<Point[][]>
moments(mat, binaryImage?: boolean): Promise<MomentsResult>
goodFeaturesToTrack(mat, maxCorners, qualityLevel, minDistance): Promise<PointF64[]>
houghLines(mat, rho, theta, threshold): Promise<HoughLine[]>
houghLinesP(mat, rho, theta, threshold, minLineLength?, maxLineGap?): Promise<LineSegment[]>
houghCircles(mat, dp, minDist, ...): Promise<Circle[]>

// Synchronous
contourArea(contour: Point[]): number
arcLength(contour: Point[], closed: boolean): number
boundingRect(contour: Point[]): Rect
minAreaRect(contour: Point[]): RotatedRect
convexHull(points: Point[]): Point[]
approxPolyDP(contour: Point[], epsilon: number, closed: boolean): Point[]
```

#### Drawing

```ts
// All drawing functions modify mat in-place synchronously and return void
drawLine(mat: Mat, pt1: Point, pt2: Point, color: Scalar, thickness?: number, lineType?: LineType): void
drawRectangle(mat: Mat, rect: Rect, color: Scalar, thickness?: number, lineType?: LineType): void
drawCircle(mat: Mat, center: Point, radius: number, color: Scalar, thickness?: number, lineType?: LineType): void
drawEllipse(mat: Mat, center: Point, axes: Size, angle: number, startAngle: number, endAngle: number, color: Scalar, thickness?: number, lineType?: LineType): void
drawContours(mat: Mat, contours: Point[][], contourIdx: number, color: Scalar, thickness?: number, lineType?: LineType): void
fillPoly(mat: Mat, pts: Point[][], color: Scalar, lineType?: LineType): void
putText(mat: Mat, text: string, org: Point, fontFace: HersheyFont, fontScale: number, color: Scalar, thickness?: number, lineType?: LineType): void
getTextSize(text: string, fontFace: HersheyFont, fontScale: number, thickness?: number): TextSize
```

#### DNN

```ts
// Load models
Net.readNetFromOnnx(path: string): Net
Net.readNetFromCaffe(protoPath: string, modelPath: string): Net
Net.readNetFromDarknet(cfgPath: string, weightsPath: string): Net

// Configure
net.setPreferableBackend(backend: DnnBackend): void
net.setPreferableTarget(target: DnnTarget): void
net.getUnconnectedOutLayersNames(): string[]

// Inference (atomically runs setInput + forward in a single lock hold)
net.run(blob: Mat, inputName?: string, outputName?: string, abortSignal?: AbortSignal): Promise<Mat>
net.runMultiple(blob: Mat, inputName: string | null, outputNames: string[], abortSignal?: AbortSignal): Promise<Mat[]>

// Preprocessing
blobFromImage(image: Mat, scaleFactor?: number, width?: number, height?: number, mean?: Scalar, swapRb?: boolean, crop?: boolean, abortSignal?: AbortSignal): Promise<Mat>
nmsBoxes(bboxes: Rect[], scores: number[], scoreThreshold: number, nmsThreshold: number, abortSignal?: AbortSignal): Promise<number[]>
```

#### Video

```ts
// Capture
const cap = VideoCapture.open('./video.mp4');   // or VideoCapture.open(0) for camera
cap.isOpened(): boolean
cap.read(): Promise<Mat | null>       // null when stream ends
cap.get(propId: number): number       // CAP_PROP_* constants
cap.set(propId: number, value: number): boolean
cap.release(): void

// Writer
const writer = VideoWriter.open('out.mp4', 'mp4v', 30, 1920, 1080);
writer.write(frame: Mat): void
writer.release(): void
```

#### Common Types

```ts
interface Point     { x: number; y: number }
interface PointF64  { x: number; y: number }
interface Rect      { x: number; y: number; width: number; height: number }
interface Size      { width: number; height: number }
interface Scalar    { v0: number; v1: number; v2: number; v3: number }
interface RotatedRect { center: PointF64; size: SizeF64; angle: number }
```

#### Constants

Matrix type constants: `CV_8U`, `CV_8UC1`, `CV_8UC2`, `CV_8UC3`, `CV_8UC4`, `CV_16U`, `CV_16S`, `CV_32S`, `CV_32F`, `CV_64F`, …

Video capture constants: `CAP_PROP_FPS`, `CAP_PROP_FRAME_WIDTH`, `CAP_PROP_FRAME_HEIGHT`, `CAP_PROP_FRAME_COUNT`, `CAP_PROP_POS_FRAMES`, `CAP_PROP_POS_MSEC`, …

Enums: `ColorCode`, `ImreadFlag`, `ThresholdType`, `InterpolationFlag`, `MorphType`, `MorphShape`, `BorderType`, `TemplateMatchMode`, `FlipCode`, `LineType`, `HersheyFont`, `NormType`, `DnnBackend`, `DnnTarget`, `ContourRetrievalMode`, `ContourApproximation`, `AdaptiveThresholdType`

### Building from Source

**1. Install system dependencies**

```bash
# Ubuntu / Debian
sudo apt-get install cmake libopencv-dev llvm clang libclang-dev

# macOS
brew install opencv llvm

# Windows — install OpenCV 4, then set:
# OPENCV_INCLUDE_PATHS, OPENCV_LINK_LIBS, OPENCV_LINK_PATHS
```

**2. Install Rust**

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**3. Clone and build**

```bash
git clone https://github.com/luckyyyyy/node-opencv.git
cd node-opencv-rs
npm install
npm run build       # release build (slower, optimised)
# or
npm run build:debug # debug build (faster, for development)
npm test
```

> On Linux you may need to set `LIBCLANG_PATH`:
> ```bash
> export LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu
> ```

### Contributing

Contributions are welcome! Please open an issue or pull request.

- Keep async operations on `napi::AsyncTask` — **no** `tokio::spawn`.
- Avoid unnecessary `.clone()` calls.
- Run `npm run build:debug && npm test` before submitting.

---

## License

[MIT](./LICENSE)
