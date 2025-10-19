# node-opencv

High-performance Node.js bindings for OpenCV using napi-rs and opencv-rust.

## Features

This library provides a comprehensive set of OpenCV functions for Node.js applications, with support for:

- Core operations (flip, rotate, merge, split, in_range)
- Image processing (Canny, Gaussian blur, thresholding, contour detection)
- Template matching
- Deep learning (DNN module support)
- Asynchronous operations using Node.js worker threads

## Installation

```bash
npm install node-opencv
```

## Usage

### Basic Example

```javascript
const cv = require('node-opencv');

async function processImage() {
  // Load an image
  const image = await cv.imread('./image.jpg');
  
  // Apply Gaussian blur
  const blurred = await cv.gaussianBlur(image, 5, 5, 1.5);
  
  // Detect edges using Canny
  const edges = await cv.canny(blurred, 50, 150);
  
  // Find contours
  const contours = await cv.findContours(
    edges,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
  );
  
  // Draw contours on the original image
  const result = await cv.drawContours(
    image,
    contours,
    -1, // Draw all contours
    { val0: 0, val1: 255, val2: 0, val3: 255 }, // Green color
    2 // Thickness
  );
  
  // Save the result
  const buffer = await cv.imencode('.jpg', result);
  require('fs').writeFileSync('./output.jpg', buffer);
}

processImage().catch(console.error);
```

## API Reference

### Image I/O

#### `imread(path: string, flags?: number): Promise<Mat>`

Read an image from a file.

```javascript
const image = await cv.imread('./image.jpg', cv.IMREAD_COLOR);
```

#### `imdecode(buffer: Buffer, flags?: number): Promise<Mat>`

Decode an image from a buffer.

```javascript
const buffer = fs.readFileSync('./image.jpg');
const image = await cv.imdecode(buffer, cv.IMREAD_COLOR);
```

#### `imencode(ext: string, mat: Mat): Promise<Buffer>`

Encode an image to a buffer.

```javascript
const buffer = await cv.imencode('.jpg', image);
```

### Core Functions

#### `flip(src: Mat, flipCode: number): Promise<Mat>`

Flip an image horizontally, vertically, or both.

- `flipCode = 0`: flip vertically
- `flipCode > 0`: flip horizontally
- `flipCode < 0`: flip both horizontally and vertically

```javascript
const flipped = await cv.flip(image, 1); // Flip horizontally
```

#### `rotate(src: Mat, rotateCode: number): Promise<Mat>`

Rotate an image by 90, 180, or 270 degrees.

```javascript
const rotated = await cv.rotate(image, cv.ROTATE_90_CLOCKWISE);
```

**Constants:**
- `cv.ROTATE_90_CLOCKWISE`
- `cv.ROTATE_180`
- `cv.ROTATE_90_COUNTERCLOCKWISE`

#### `split(src: Mat): Promise<Mat[]>`

Split a multi-channel array into several single-channel arrays.

```javascript
const [blue, green, red] = await cv.split(image);
```

#### `merge(mats: Mat[]): Promise<Mat>`

Merge several single-channel arrays into a multi-channel array.

```javascript
const merged = await cv.merge([blue, green, red]);
```

#### `inRange(src: Mat, lowerBound: number[], upperBound: number[]): Promise<Mat>`

Check if array elements lie between bounds.

```javascript
// Create a mask for pixels in the range [0, 0, 0] to [128, 128, 128]
const mask = await cv.inRange(image, [0, 0, 0], [128, 128, 128]);
```

### Image Processing

#### `canny(src: Mat, threshold1: number, threshold2: number, apertureSize?: number, l2Gradient?: boolean): Promise<Mat>`

Find edges in an image using the Canny algorithm.

```javascript
const edges = await cv.canny(image, 50, 150);
```

#### `gaussianBlur(src: Mat, ksizeWidth: number, ksizeHeight: number, sigmaX: number, sigmaY?: number): Promise<Mat>`

Apply Gaussian blur to an image.

```javascript
const blurred = await cv.gaussianBlur(image, 5, 5, 1.5);
```

#### `threshold(src: Mat, thresh: number, maxval: number, type: number): Promise<Mat>`

Apply a fixed-level threshold to an image.

```javascript
const thresholded = await image.threshold(127, 255, cv.THRESH_BINARY);
```

**Threshold types:**
- `cv.THRESH_BINARY`
- `cv.THRESH_BINARY_INV`
- `cv.THRESH_TRUNC`
- `cv.THRESH_TOZERO`
- `cv.THRESH_TOZERO_INV`
- `cv.THRESH_OTSU`
- `cv.THRESH_TRIANGLE`

#### `adaptiveThreshold(src: Mat, maxValue: number, adaptiveMethod: number, thresholdType: number, blockSize: number, c: number): Promise<Mat>`

Apply adaptive threshold to an image.

```javascript
const adaptive = await cv.adaptiveThreshold(
  grayImage,
  255,
  cv.ADAPTIVE_THRESH_GAUSSIAN_C,
  cv.THRESH_BINARY,
  11,
  2
);
```

**Adaptive methods:**
- `cv.ADAPTIVE_THRESH_MEAN_C`
- `cv.ADAPTIVE_THRESH_GAUSSIAN_C`

#### `findContours(src: Mat, mode: number, method: number): Promise<Contour[]>`

Find contours in a binary image.

```javascript
const contours = await cv.findContours(
  binaryImage,
  cv.RETR_EXTERNAL,
  cv.CHAIN_APPROX_SIMPLE
);
```

**Retrieval modes:**
- `cv.RETR_EXTERNAL` - retrieves only the extreme outer contours
- `cv.RETR_LIST` - retrieves all contours without hierarchy
- `cv.RETR_CCOMP` - retrieves contours with 2-level hierarchy
- `cv.RETR_TREE` - retrieves all contours with full hierarchy

**Approximation methods:**
- `cv.CHAIN_APPROX_NONE` - stores all contour points
- `cv.CHAIN_APPROX_SIMPLE` - compresses horizontal, vertical, and diagonal segments
- `cv.CHAIN_APPROX_TC89_L1`
- `cv.CHAIN_APPROX_TC89_KCOS`

#### `drawContours(src: Mat, contours: Contour[], contourIdx: number, color: Scalar, thickness?: number): Promise<Mat>`

Draw contours on an image.

```javascript
const result = await cv.drawContours(
  image,
  contours,
  -1, // Draw all contours
  { val0: 0, val1: 255, val2: 0, val3: 255 }, // Green in BGRA
  2
);
```

### Template Matching

#### `matchTemplate(template: Mat, method: number): Promise<Mat>`

Match a template within an image.

```javascript
const result = await image.matchTemplate(template, cv.TM_CCOEFF_NORMED);
```

**Methods:**
- `cv.TM_SQDIFF`
- `cv.TM_SQDIFF_NORMED`
- `cv.TM_CCORR`
- `cv.TM_CCORR_NORMED`
- `cv.TM_CCOEFF`
- `cv.TM_CCOEFF_NORMED`

#### `matchTemplateAll(template: Mat, method: number, score: number, nmsThreshold: number): Promise<Rect[]>`

Find all template matches above a threshold.

```javascript
const matches = await image.matchTemplateAll(
  template,
  cv.TM_CCOEFF_NORMED,
  0.8, // Minimum score
  0.1  // NMS threshold
);
```

### Mat Methods

#### `mat.rows: number`

Get the number of rows in the matrix.

#### `mat.cols: number`

Get the number of columns in the matrix.

#### `mat.size: Size`

Get the size of the matrix.

```javascript
const { width, height } = image.size;
```

#### `mat.data: Buffer`

Get the raw pixel data as a Buffer.

```javascript
const pixels = image.data;
```

#### `mat.release(): void`

Release the matrix memory (optional, memory is managed automatically).

## TypeScript Support

This library includes TypeScript definitions for all functions and types:

```typescript
import * as cv from 'node-opencv';

async function processImage(): Promise<void> {
  const image: cv.Mat = await cv.imread('./image.jpg');
  const edges: cv.Mat = await cv.canny(image, 50, 150);
  const contours: cv.Contour[] = await cv.findContours(
    edges,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
  );
}
```

## Performance

All operations are asynchronous and execute on worker threads, ensuring the main event loop remains responsive. This makes the library suitable for high-performance applications and servers.

## License

MIT
