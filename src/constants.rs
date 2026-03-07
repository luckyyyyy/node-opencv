// ─── Template match modes ─────────────────────────────────────────────────────

#[napi]
pub enum TemplateMatchMode {
  Sqdiff = 0,
  SqdiffNormed = 1,
  Ccorr = 2,
  CcorrNormed = 3,
  Ccoeff = 4,
  CcoeffNormed = 5,
}

// ─── Imread flags ─────────────────────────────────────────────────────────────

#[napi]
pub enum ImreadFlag {
  Unchanged = -1,
  Grayscale = 0,
  Color = 1,
  AnyDepth = 2,
  AnyColor = 4,
  LoadGdal = 8,
  ReducedGrayscale2 = 16,
  ReducedColor2 = 17,
  ReducedGrayscale4 = 32,
  ReducedColor4 = 33,
  ReducedGrayscale8 = 64,
  ReducedColor8 = 65,
  IgnoreOrientation = 128,
}

// ─── Color conversion codes ───────────────────────────────────────────────────

#[napi]
pub enum ColorCode {
  Bgr2Bgra = 0,
  Bgra2Bgr = 1,
  Bgr2Rgba = 2,
  Rgba2Bgr = 3,
  Bgr2Rgb = 4,
  Bgra2Rgba = 5,
  Bgr2Gray = 6,
  Rgb2Gray = 7,
  Gray2Bgr = 8,
  Gray2Bgra = 9,
  Bgr2Xyz = 32,
  Xyz2Bgr = 34,
  Bgr2YCrCb = 36,
  YCrCb2Bgr = 38,
  Bgr2Hsv = 40,
  Bgr2Lab = 44,
  Bgr2Hls = 52,
  Hsv2Bgr = 54,
  Lab2Bgr = 56,
  Hls2Bgr = 60,
}

// ─── Threshold types ──────────────────────────────────────────────────────────

#[napi]
pub enum ThresholdType {
  Binary = 0,
  BinaryInv = 1,
  Trunc = 2,
  ToZero = 3,
  ToZeroInv = 4,
  Otsu = 8,
  Triangle = 16,
}

// ─── Adaptive threshold methods ───────────────────────────────────────────────

#[napi]
pub enum AdaptiveThresholdType {
  MeanC = 0,
  GaussianC = 1,
}

// ─── Morphology operation types ───────────────────────────────────────────────

#[napi]
pub enum MorphType {
  Erode = 0,
  Dilate = 1,
  Open = 2,
  Close = 3,
  Gradient = 4,
  TopHat = 5,
  BlackHat = 6,
  HitMiss = 7,
}

// ─── Morphology kernel shapes ─────────────────────────────────────────────────

#[napi]
pub enum MorphShape {
  Rect = 0,
  Cross = 1,
  Ellipse = 2,
}

// ─── Hough transform methods ──────────────────────────────────────────────────

#[napi]
pub enum HoughMethod {
  Gradient = 3,
  GradientAlt = 4,
}

// ─── Interpolation flags ──────────────────────────────────────────────────────

#[napi]
pub enum InterpolationFlag {
  Nearest = 0,
  Linear = 1,
  Cubic = 2,
  Area = 3,
  Lanczos4 = 4,
  LinearExact = 5,
  NearestExact = 6,
}

// ─── Border types ─────────────────────────────────────────────────────────────
// Note: BORDER_DEFAULT == BORDER_REFLECT_101 (both = 4) — use Reflect101.

#[napi]
pub enum BorderType {
  Constant = 0,
  Replicate = 1,
  Reflect = 2,
  Wrap = 3,
  Reflect101 = 4,
  Transparent = 5,
  Isolated = 16,
}

// ─── Norm types ───────────────────────────────────────────────────────────────

#[napi]
pub enum NormType {
  Inf = 1,
  L1 = 2,
  L2 = 4,
  L2Sqr = 5,
  Hamming = 6,
  Hamming2 = 7,
  MinMax = 32,
}

// ─── Mat type codes (kept as flat consts — numeric depth/type parameters) ─────

#[napi]
pub const CV_8U: i32 = opencv::core::CV_8U;
#[napi]
pub const CV_8S: i32 = opencv::core::CV_8S;
#[napi]
pub const CV_16U: i32 = opencv::core::CV_16U;
#[napi]
pub const CV_16S: i32 = opencv::core::CV_16S;
#[napi]
pub const CV_32S: i32 = opencv::core::CV_32S;
#[napi]
pub const CV_32F: i32 = opencv::core::CV_32F;
#[napi]
pub const CV_64F: i32 = opencv::core::CV_64F;
#[napi]
pub const CV_8UC1: i32 = opencv::core::CV_8UC1;
#[napi]
pub const CV_8UC2: i32 = opencv::core::CV_8UC2;
#[napi]
pub const CV_8UC3: i32 = opencv::core::CV_8UC3;
#[napi]
pub const CV_8UC4: i32 = opencv::core::CV_8UC4;
#[napi]
pub const CV_16UC1: i32 = opencv::core::CV_16UC1;
#[napi]
pub const CV_16UC3: i32 = opencv::core::CV_16UC3;
#[napi]
pub const CV_16SC1: i32 = opencv::core::CV_16SC1;
#[napi]
pub const CV_16SC3: i32 = opencv::core::CV_16SC3;
#[napi]
pub const CV_32SC1: i32 = opencv::core::CV_32SC1;
#[napi]
pub const CV_32FC1: i32 = opencv::core::CV_32FC1;
#[napi]
pub const CV_32FC3: i32 = opencv::core::CV_32FC3;
#[napi]
pub const CV_64FC1: i32 = opencv::core::CV_64FC1;
#[napi]
pub const CV_64FC3: i32 = opencv::core::CV_64FC3;

// ─── Line types ───────────────────────────────────────────────────────────────

#[napi]
pub enum LineType {
  Line4 = 4,
  Line8 = 8,
  AntiAlias = 16,
}

// ─── Hershey fonts ────────────────────────────────────────────────────────────

#[napi]
pub enum HersheyFont {
  // ─ base fonts ───────────────────────────────────────────────────
  Simplex = 0,
  Plain = 1,
  Duplex = 2,
  Complex = 3,
  Triplex = 4,
  ComplexSmall = 5,
  ScriptSimplex = 6,
  ScriptComplex = 7,
  // ─ italic variants (base | FONT_ITALIC) ──────────────────────────
  SimplexItalic = 16,
  PlainItalic = 17,
  DuplexItalic = 18,
  ComplexItalic = 19,
  TriplexItalic = 20,
  ComplexSmallItalic = 21,
  ScriptSimplexItalic = 22,
  ScriptComplexItalic = 23,
}

#[napi]
pub const FONT_ITALIC: i32 = opencv::imgproc::FONT_ITALIC;

// ─── Contour retrieval modes ──────────────────────────────────────────────────

#[napi]
pub enum ContourRetrievalMode {
  External = 0,
  List = 1,
  CComp = 2,
  Tree = 3,
}

// ─── Contour approximation methods ───────────────────────────────────────────

#[napi]
pub enum ContourApproximation {
  None = 1,
  Simple = 2,
  Tc89L1 = 3,
  Tc89Kcos = 4,
}

// ─── Warp flags (kept as consts — bitwise-OR'd with InterpolationFlag) ────────

#[napi]
pub const WARP_FILL_OUTLIERS: i32 = opencv::imgproc::WARP_FILL_OUTLIERS;
#[napi]
pub const WARP_INVERSE_MAP: i32 = opencv::imgproc::WARP_INVERSE_MAP;

// ─── Flip codes ───────────────────────────────────────────────────────────────

#[napi]
pub enum FlipCode {
  Vertical = 0,
  Horizontal = 1,
  Both = -1,
}

// ─── DNN backends ─────────────────────────────────────────────────────────────

#[napi]
pub enum DnnBackend {
  Default = 0,
  Halide = 1,
  InferenceEngine = 2,
  OpenCv = 3,
  Vkcom = 4,
  Cuda = 5,
  Webnn = 6,
  Timvx = 7,
}

// ─── DNN targets ──────────────────────────────────────────────────────────────

#[napi]
pub enum DnnTarget {
  Cpu = 0,
  OpenCl = 1,
  OpenClFp16 = 2,
  Myriad = 3,
  Vulkan = 4,
  Fpga = 5,
  Cuda = 6,
  CudaFp16 = 7,
  Hddl = 8,
  Npu = 9,
}

// ─── VideoCapture property IDs (kept as consts — arbitrary numeric IDs) ───────

#[napi]
pub const CAP_PROP_POS_MSEC: i32 = opencv::videoio::CAP_PROP_POS_MSEC;
#[napi]
pub const CAP_PROP_POS_FRAMES: i32 = opencv::videoio::CAP_PROP_POS_FRAMES;
#[napi]
pub const CAP_PROP_FRAME_WIDTH: i32 = opencv::videoio::CAP_PROP_FRAME_WIDTH;
#[napi]
pub const CAP_PROP_FRAME_HEIGHT: i32 = opencv::videoio::CAP_PROP_FRAME_HEIGHT;
#[napi]
pub const CAP_PROP_FPS: i32 = opencv::videoio::CAP_PROP_FPS;
#[napi]
pub const CAP_PROP_FOURCC: i32 = opencv::videoio::CAP_PROP_FOURCC;
#[napi]
pub const CAP_PROP_FRAME_COUNT: i32 = opencv::videoio::CAP_PROP_FRAME_COUNT;
#[napi]
pub const CAP_PROP_FORMAT: i32 = opencv::videoio::CAP_PROP_FORMAT;
#[napi]
pub const CAP_PROP_BRIGHTNESS: i32 = opencv::videoio::CAP_PROP_BRIGHTNESS;
#[napi]
pub const CAP_PROP_CONTRAST: i32 = opencv::videoio::CAP_PROP_CONTRAST;
#[napi]
pub const CAP_PROP_BITRATE: i32 = opencv::videoio::CAP_PROP_BITRATE;
#[napi]
pub const CAP_PROP_BACKEND: i32 = opencv::videoio::CAP_PROP_BACKEND;

// ─── Compile-time assertions: verify enum values match OpenCV constants ────────

const _: () = {
  assert!(TemplateMatchMode::Sqdiff as i32 == opencv::imgproc::TM_SQDIFF);
  assert!(TemplateMatchMode::CcoeffNormed as i32 == opencv::imgproc::TM_CCOEFF_NORMED);
  assert!(ImreadFlag::Unchanged as i32 == opencv::imgcodecs::IMREAD_UNCHANGED);
  assert!(ImreadFlag::Color as i32 == opencv::imgcodecs::IMREAD_COLOR);
  assert!(ColorCode::Bgr2Gray as i32 == opencv::imgproc::COLOR_BGR2GRAY);
  assert!(ColorCode::Bgr2Hsv as i32 == opencv::imgproc::COLOR_BGR2HSV);
  assert!(ThresholdType::Binary as i32 == opencv::imgproc::THRESH_BINARY);
  assert!(ThresholdType::Otsu as i32 == opencv::imgproc::THRESH_OTSU);
  assert!(AdaptiveThresholdType::MeanC as i32 == opencv::imgproc::ADAPTIVE_THRESH_MEAN_C);
  assert!(MorphType::Open as i32 == opencv::imgproc::MORPH_OPEN);
  assert!(MorphShape::Rect as i32 == opencv::imgproc::MORPH_RECT);
  assert!(InterpolationFlag::Linear as i32 == opencv::imgproc::INTER_LINEAR);
  assert!(BorderType::Constant as i32 == opencv::core::BORDER_CONSTANT);
  assert!(BorderType::Reflect101 as i32 == opencv::core::BORDER_REFLECT_101);
  assert!(NormType::MinMax as i32 == opencv::core::NORM_MINMAX);
  assert!(LineType::Line8 as i32 == opencv::imgproc::LINE_8);
  assert!(HersheyFont::Simplex as i32 == opencv::imgproc::FONT_HERSHEY_SIMPLEX);
  assert!(HersheyFont::ComplexSmall as i32 == opencv::imgproc::FONT_HERSHEY_COMPLEX_SMALL);
  assert!(HersheyFont::ScriptSimplex as i32 == opencv::imgproc::FONT_HERSHEY_SCRIPT_SIMPLEX);
  assert!(HersheyFont::ScriptComplex as i32 == opencv::imgproc::FONT_HERSHEY_SCRIPT_COMPLEX);
  assert!(FONT_ITALIC == opencv::imgproc::FONT_ITALIC);
  assert!(ContourRetrievalMode::External as i32 == opencv::imgproc::RETR_EXTERNAL);
  assert!(ContourApproximation::Simple as i32 == opencv::imgproc::CHAIN_APPROX_SIMPLE);
  assert!(DnnBackend::OpenCv as i32 == opencv::dnn::DNN_BACKEND_OPENCV);
  assert!(DnnTarget::Cpu as i32 == opencv::dnn::DNN_TARGET_CPU);
};
