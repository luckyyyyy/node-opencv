use crate::{JSMat, OpenCVError};
use napi::bindgen_prelude::*;
use opencv::{core as cv_core, imgproc};

#[napi(object)]
#[derive(Clone)]
pub struct Scalar {
  pub val0: f64,
  pub val1: f64,
  pub val2: f64,
  pub val3: f64,
}

/// Canny edge detection task
pub struct CannyTask {
  source_mat: cv_core::Mat,
  threshold1: f64,
  threshold2: f64,
  aperture_size: i32,
  l2_gradient: bool,
}

#[napi]
impl Task for CannyTask {
  type Output = JSMat;
  type JsValue = JSMat;

  fn compute(&mut self) -> Result<Self::Output> {
    let mut dst = cv_core::Mat::default();
    imgproc::canny(
      &self.source_mat,
      &mut dst,
      self.threshold1,
      self.threshold2,
      self.aperture_size,
      self.l2_gradient,
    )
    .map_err(OpenCVError)?;
    Ok(JSMat { mat: dst })
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

/// Gaussian blur task
pub struct GaussianBlurTask {
  source_mat: cv_core::Mat,
  ksize_width: i32,
  ksize_height: i32,
  sigma_x: f64,
  sigma_y: f64,
}

#[napi]
impl Task for GaussianBlurTask {
  type Output = JSMat;
  type JsValue = JSMat;

  fn compute(&mut self) -> Result<Self::Output> {
    let mut dst = cv_core::Mat::default();
    let ksize = cv_core::Size::new(self.ksize_width, self.ksize_height);
    imgproc::gaussian_blur(
      &self.source_mat,
      &mut dst,
      ksize,
      self.sigma_x,
      self.sigma_y,
      cv_core::BORDER_DEFAULT,
    )
    .map_err(OpenCVError)?;
    Ok(JSMat { mat: dst })
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

/// Adaptive threshold task
pub struct AdaptiveThresholdTask {
  source_mat: cv_core::Mat,
  max_value: f64,
  adaptive_method: i32,
  threshold_type: i32,
  block_size: i32,
  c: f64,
}

#[napi]
impl Task for AdaptiveThresholdTask {
  type Output = JSMat;
  type JsValue = JSMat;

  fn compute(&mut self) -> Result<Self::Output> {
    let mut dst = cv_core::Mat::default();
    imgproc::adaptive_threshold(
      &self.source_mat,
      &mut dst,
      self.max_value,
      self.adaptive_method,
      self.threshold_type,
      self.block_size,
      self.c,
    )
    .map_err(OpenCVError)?;
    Ok(JSMat { mat: dst })
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

/// Find contours task
pub struct FindContoursTask {
  source_mat: cv_core::Mat,
  mode: i32,
  method: i32,
}

#[napi(object)]
#[derive(Clone)]
pub struct Point {
  pub x: i32,
  pub y: i32,
}

pub type Contour = Vec<Point>;

#[napi]
impl Task for FindContoursTask {
  type Output = Vec<Contour>;
  type JsValue = Vec<Contour>;

  fn compute(&mut self) -> Result<Self::Output> {
    let mut contours = cv_core::Vector::<cv_core::Vector<cv_core::Point>>::new();
    let mut hierarchy = cv_core::Mat::default();
    imgproc::find_contours_with_hierarchy(
      &self.source_mat,
      &mut contours,
      &mut hierarchy,
      self.mode,
      self.method,
      cv_core::Point::new(0, 0),
    )
    .map_err(OpenCVError)?;

    let mut result: Vec<Contour> = Vec::new();
    for i in 0..contours.len() {
      let contour = contours.get(i).map_err(OpenCVError)?;
      let points: Vec<Point> = contour
        .iter()
        .map(|p| Point { x: p.x, y: p.y })
        .collect();
      result.push(points);
    }
    Ok(result)
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

/// Draw contours task
pub struct DrawContoursTask {
  source_mat: cv_core::Mat,
  contours: Vec<Contour>,
  contour_idx: i32,
  color: cv_core::Scalar,
  thickness: i32,
}

#[napi]
impl Task for DrawContoursTask {
  type Output = JSMat;
  type JsValue = JSMat;

  fn compute(&mut self) -> Result<Self::Output> {
    let mut dst = self.source_mat.clone();
    let mut opencv_contours = cv_core::Vector::<cv_core::Vector<cv_core::Point>>::new();
    
    for contour in &self.contours {
      let mut opencv_contour = cv_core::Vector::<cv_core::Point>::new();
      for point in contour {
        opencv_contour.push(cv_core::Point::new(point.x, point.y));
      }
      opencv_contours.push(opencv_contour);
    }

    imgproc::draw_contours(
      &mut dst,
      &opencv_contours,
      self.contour_idx,
      self.color,
      self.thickness,
      imgproc::LINE_8,
      &cv_core::Mat::default(),
      i32::MAX,
      cv_core::Point::new(0, 0),
    )
    .map_err(OpenCVError)?;
    Ok(JSMat { mat: dst })
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

/// Find edges in an image using the Canny algorithm.
/// 
/// # Arguments
/// * `src` - Source image (single-channel 8-bit)
/// * `threshold1` - First threshold for the hysteresis procedure
/// * `threshold2` - Second threshold for the hysteresis procedure
/// * `aperture_size` - Aperture size for the Sobel operator (default: 3)
/// * `l2_gradient` - Flag indicating whether to use L2 norm for gradient (default: false)
#[napi]
pub fn canny(
  #[napi(ts_arg_type = "JSMat")] src: &JSMat,
  threshold1: f64,
  threshold2: f64,
  aperture_size: Option<i32>,
  l2_gradient: Option<bool>,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<CannyTask> {
  AsyncTask::with_optional_signal(
    CannyTask {
      source_mat: src.mat.clone(),
      threshold1,
      threshold2,
      aperture_size: aperture_size.unwrap_or(3),
      l2_gradient: l2_gradient.unwrap_or(false),
    },
    abort_signal,
  )
}

/// Apply Gaussian blur to an image.
/// 
/// # Arguments
/// * `src` - Source image
/// * `ksize_width` - Gaussian kernel width (must be positive and odd)
/// * `ksize_height` - Gaussian kernel height (must be positive and odd)
/// * `sigma_x` - Gaussian kernel standard deviation in X direction
/// * `sigma_y` - Gaussian kernel standard deviation in Y direction (default: 0, uses sigma_x)
#[napi]
pub fn gaussian_blur(
  #[napi(ts_arg_type = "JSMat")] src: &JSMat,
  ksize_width: i32,
  ksize_height: i32,
  sigma_x: f64,
  sigma_y: Option<f64>,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<GaussianBlurTask> {
  AsyncTask::with_optional_signal(
    GaussianBlurTask {
      source_mat: src.mat.clone(),
      ksize_width,
      ksize_height,
      sigma_x,
      sigma_y: sigma_y.unwrap_or(0.0),
    },
    abort_signal,
  )
}

/// Apply adaptive threshold to an array.
/// 
/// # Arguments
/// * `src` - Source 8-bit single-channel image
/// * `max_value` - Non-zero value assigned to pixels for which the condition is satisfied
/// * `adaptive_method` - Adaptive thresholding algorithm (ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C)
/// * `threshold_type` - Thresholding type (THRESH_BINARY or THRESH_BINARY_INV)
/// * `block_size` - Size of pixel neighborhood used to calculate threshold (must be odd and greater than 1)
/// * `c` - Constant subtracted from the mean or weighted mean
#[napi]
pub fn adaptive_threshold(
  #[napi(ts_arg_type = "JSMat")] src: &JSMat,
  max_value: f64,
  adaptive_method: i32,
  threshold_type: i32,
  block_size: i32,
  c: f64,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<AdaptiveThresholdTask> {
  AsyncTask::with_optional_signal(
    AdaptiveThresholdTask {
      source_mat: src.mat.clone(),
      max_value,
      adaptive_method,
      threshold_type,
      block_size,
      c,
    },
    abort_signal,
  )
}

/// Find contours in a binary image.
/// 
/// # Arguments
/// * `src` - Source 8-bit single-channel image (non-zero pixels are treated as 1s)
/// * `mode` - Contour retrieval mode (RETR_EXTERNAL, RETR_LIST, RETR_CCOMP, RETR_TREE)
/// * `method` - Contour approximation method (CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE, etc.)
#[napi]
pub fn find_contours(
  #[napi(ts_arg_type = "JSMat")] src: &JSMat,
  mode: i32,
  method: i32,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<FindContoursTask> {
  AsyncTask::with_optional_signal(
    FindContoursTask {
      source_mat: src.mat.clone(),
      mode,
      method,
    },
    abort_signal,
  )
}

/// Draw contours on an image.
/// 
/// # Arguments
/// * `src` - Destination image
/// * `contours` - All input contours (array of point arrays)
/// * `contour_idx` - Contour to draw (-1 means all contours)
/// * `color` - Color of the contours (Scalar with val0, val1, val2, val3)
/// * `thickness` - Thickness of lines the contours are drawn with (negative value fills the contour)
#[napi]
pub fn draw_contours(
  #[napi(ts_arg_type = "JSMat")] src: &JSMat,
  contours: Vec<Contour>,
  contour_idx: i32,
  color: Scalar,
  thickness: Option<i32>,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<DrawContoursTask> {
  let opencv_color = cv_core::Scalar::new(color.val0, color.val1, color.val2, color.val3);
  AsyncTask::with_optional_signal(
    DrawContoursTask {
      source_mat: src.mat.clone(),
      contours,
      contour_idx,
      color: opencv_color,
      thickness: thickness.unwrap_or(1),
    },
    abort_signal,
  )
}
