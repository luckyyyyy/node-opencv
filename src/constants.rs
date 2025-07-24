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

// Flip codes for cv::core::flip
#[napi]
pub const FLIP_HORIZONTAL: i32 = 1;

#[napi]
pub const FLIP_VERTICAL: i32 = 0;

#[napi]
pub const FLIP_BOTH: i32 = -1;
