#[napi]
pub const TM_SQDIFF: i32 = opencv::imgproc::TM_SQDIFF;

#[napi]
pub const TM_SQDIFF_NORMED: i32 = opencv::imgproc::TM_SQDIFF_NORMED;

#[napi]
pub const TM_CCORR: i32 = opencv::imgproc::TM_CCORR;

#[napi]
pub const TM_CCORR_NORMED: i32 = opencv::imgproc::TM_CCORR_NORMED;

#[napi]
pub const TM_CCOEFF: i32 = opencv::imgproc::TM_CCOEFF;

#[napi]
pub const TM_CCOEFF_NORMED: i32 = opencv::imgproc::TM_CCOEFF_NORMED;

#[napi]
pub const IMREAD_UNCHANGED: i32 = opencv::imgcodecs::IMREAD_UNCHANGED;

#[napi]
pub const IMREAD_GRAYSCALE: i32 = opencv::imgcodecs::IMREAD_GRAYSCALE;

#[napi]
pub const IMREAD_COLOR: i32 = opencv::imgcodecs::IMREAD_COLOR;

#[napi]
pub const IMREAD_ANYDEPTH: i32 = opencv::imgcodecs::IMREAD_ANYDEPTH;

#[napi]
pub const IMREAD_ANYCOLOR: i32 = opencv::imgcodecs::IMREAD_ANYCOLOR;

#[napi]
pub const IMREAD_LOAD_GDAL: i32 = opencv::imgcodecs::IMREAD_LOAD_GDAL;

#[napi]
pub const IMREAD_REDUCED_GRAYSCALE_2: i32 = opencv::imgcodecs::IMREAD_REDUCED_GRAYSCALE_2;

#[napi]
pub const IMREAD_REDUCED_COLOR_2: i32 = opencv::imgcodecs::IMREAD_REDUCED_COLOR_2;

#[napi]
pub const IMREAD_REDUCED_GRAYSCALE_4: i32 = opencv::imgcodecs::IMREAD_REDUCED_GRAYSCALE_4;

#[napi]
pub const IMREAD_REDUCED_COLOR_4: i32 = opencv::imgcodecs::IMREAD_REDUCED_COLOR_4;

#[napi]
pub const IMREAD_REDUCED_GRAYSCALE_8: i32 = opencv::imgcodecs::IMREAD_REDUCED_GRAYSCALE_8;

#[napi]
pub const IMREAD_REDUCED_COLOR_8: i32 = opencv::imgcodecs::IMREAD_REDUCED_COLOR_8;

#[napi]
pub const IMREAD_IGNORE_ORIENTATION: i32 = opencv::imgcodecs::IMREAD_IGNORE_ORIENTATION;

// Threshold types
#[napi]
pub const THRESH_BINARY: i32 = opencv::imgproc::THRESH_BINARY;

#[napi]
pub const THRESH_BINARY_INV: i32 = opencv::imgproc::THRESH_BINARY_INV;

#[napi]
pub const THRESH_TRUNC: i32 = opencv::imgproc::THRESH_TRUNC;

#[napi]
pub const THRESH_TOZERO: i32 = opencv::imgproc::THRESH_TOZERO;

#[napi]
pub const THRESH_TOZERO_INV: i32 = opencv::imgproc::THRESH_TOZERO_INV;

#[napi]
pub const THRESH_OTSU: i32 = opencv::imgproc::THRESH_OTSU;

#[napi]
pub const THRESH_TRIANGLE: i32 = opencv::imgproc::THRESH_TRIANGLE;

// Adaptive threshold methods
#[napi]
pub const ADAPTIVE_THRESH_MEAN_C: i32 = opencv::imgproc::ADAPTIVE_THRESH_MEAN_C;

#[napi]
pub const ADAPTIVE_THRESH_GAUSSIAN_C: i32 = opencv::imgproc::ADAPTIVE_THRESH_GAUSSIAN_C;

// Contour retrieval modes
#[napi]
pub const RETR_EXTERNAL: i32 = opencv::imgproc::RETR_EXTERNAL;

#[napi]
pub const RETR_LIST: i32 = opencv::imgproc::RETR_LIST;

#[napi]
pub const RETR_CCOMP: i32 = opencv::imgproc::RETR_CCOMP;

#[napi]
pub const RETR_TREE: i32 = opencv::imgproc::RETR_TREE;

// Contour approximation methods
#[napi]
pub const CHAIN_APPROX_NONE: i32 = opencv::imgproc::CHAIN_APPROX_NONE;

#[napi]
pub const CHAIN_APPROX_SIMPLE: i32 = opencv::imgproc::CHAIN_APPROX_SIMPLE;

#[napi]
pub const CHAIN_APPROX_TC89_L1: i32 = opencv::imgproc::CHAIN_APPROX_TC89_L1;

#[napi]
pub const CHAIN_APPROX_TC89_KCOS: i32 = opencv::imgproc::CHAIN_APPROX_TC89_KCOS;

// Rotate codes
#[napi]
pub const ROTATE_90_CLOCKWISE: i32 = opencv::core::ROTATE_90_CLOCKWISE as i32;

#[napi]
pub const ROTATE_180: i32 = opencv::core::ROTATE_180 as i32;

#[napi]
pub const ROTATE_90_COUNTERCLOCKWISE: i32 = opencv::core::ROTATE_90_COUNTERCLOCKWISE as i32;
