use crate::constants::{
  AdaptiveThresholdType, ContourApproximation, ContourRetrievalMode, HoughMethod, MorphShape,
  ThresholdType,
};
use crate::cv_err;
use crate::impl_mat_task;
use crate::impl_passthrough_task;
use crate::mat::JSMat;
use crate::types::{pts_to_cv, Point, PointF64, Rect, RotatedRect};
use napi::bindgen_prelude::*;
use opencv::{
  core::{Mat, ToInputArray},
  imgproc,
  prelude::*,
};
use std::sync::Arc;

// ─── FindContoursTask ────────────────────────────────────────────────────────

pub(crate) struct FindContoursTask {
  src: Arc<Mat>,
  mode: i32,
  method: i32,
}

impl FindContoursTask {
  fn do_compute(&mut self) -> Result<Vec<Vec<Point>>> {
    let mut contours: opencv::core::Vector<opencv::core::Vector<opencv::core::Point>> =
      opencv::core::Vector::new();
    cv_err!(imgproc::find_contours(
      &*self.src,
      &mut contours,
      self.mode,
      self.method,
      opencv::core::Point::default(),
    ))?;
    Ok(
      contours
        .into_iter()
        .map(|c| c.into_iter().map(Into::into).collect())
        .collect(),
    )
  }
}
impl_passthrough_task!(FindContoursTask, Vec<Vec<Point>>);

// ─── contourArea (sync) ──────────────────────────────────────────────────────

#[napi(js_name = "contourArea")]
pub fn contour_area(contour: Vec<Point>) -> Result<f64> {
  let pts = pts_to_cv(&contour);
  cv_err!(imgproc::contour_area(&pts, false))
}

// ─── boundingRect (sync) ─────────────────────────────────────────────────────

#[napi(js_name = "boundingRect")]
pub fn bounding_rect(contour: Vec<Point>) -> Result<Rect> {
  let pts = pts_to_cv(&contour);
  let r = cv_err!(imgproc::bounding_rect(&pts))?;
  Ok(r.into())
}

// ─── minAreaRect (sync) ──────────────────────────────────────────────────────

#[napi(js_name = "minAreaRect")]
pub fn min_area_rect(contour: Vec<Point>) -> Result<RotatedRect> {
  let pts = pts_to_cv(&contour);
  let rr = cv_err!(imgproc::min_area_rect(&pts))?;
  Ok(RotatedRect {
    center: rr.center.into(),
    size: rr.size.into(),
    angle: rr.angle as f64,
  })
}

// ─── GoodFeaturesToTrackTask ─────────────────────────────────────────────────

pub(crate) struct GoodFeaturesToTrackTask {
  src: Arc<Mat>,
  max_corners: i32,
  quality_level: f64,
  min_distance: f64,
  block_size: i32,
  use_harris: bool,
  k: f64,
  mask: Option<Arc<Mat>>,
}

impl GoodFeaturesToTrackTask {
  fn do_compute(&mut self) -> Result<Vec<PointF64>> {
    let mut corners = Mat::default();
    // Unify optional mask via `input_array()`: both Mat and no_array() yield
    // BoxedRef<'_, _InputArray>, avoiding a full 9-argument call duplication.
    let no_arr = opencv::core::no_array();
    let mask_ia = match &self.mask {
      Some(m) => cv_err!(m.as_ref().input_array())?,
      None => cv_err!(no_arr.input_array())?,
    };
    cv_err!(imgproc::good_features_to_track(
      &*self.src,
      &mut corners,
      self.max_corners,
      self.quality_level,
      self.min_distance,
      &mask_ia,
      self.block_size,
      self.use_harris,
      self.k,
    ))?;
    // good_features_to_track returns Nx1 CV_32FC2; Vec2f layout matches Point2f.
    // When no corners are found, OpenCV may leave the output as a default empty
    // Mat (CV_8UC1), so guard with an early return before calling data_typed.
    if corners.empty() {
      return Ok(Vec::new());
    }
    let data = cv_err!(corners.data_typed::<opencv::core::Vec2f>())?;
    Ok(
      data
        .iter()
        .map(|v| PointF64 {
          x: v[0] as f64,
          y: v[1] as f64,
        })
        .collect(),
    )
  }
}
impl_passthrough_task!(GoodFeaturesToTrackTask, Vec<PointF64>);

// ─── HoughCirclesTask ────────────────────────────────────────────────────────

#[napi(object)]
#[derive(Clone, Debug)]
pub struct Circle {
  pub x: f64,
  pub y: f64,
  pub radius: f64,
}

pub(crate) struct HoughCirclesTask {
  src: Arc<Mat>,
  method: i32,
  dp: f64,
  min_dist: f64,
  param1: f64,
  param2: f64,
  min_radius: i32,
  max_radius: i32,
}

impl HoughCirclesTask {
  fn do_compute(&mut self) -> Result<Vec<Circle>> {
    let mut circles = Mat::default();
    cv_err!(imgproc::hough_circles(
      &*self.src,
      &mut circles,
      self.method,
      self.dp,
      self.min_dist,
      self.param1,
      self.param2,
      self.min_radius,
      self.max_radius,
    ))?;
    // hough_circles returns 1xN CV_32FC3 (x, y, radius per element).
    // When no circles are found, OpenCV may leave the output as a default empty
    // Mat (CV_8UC1), so guard before calling data_typed to avoid a type mismatch error.
    if circles.empty() {
      return Ok(Vec::new());
    }
    let data = cv_err!(circles.data_typed::<opencv::core::Vec3f>())?;
    Ok(
      data
        .iter()
        .map(|v| Circle {
          x: v[0] as f64,
          y: v[1] as f64,
          radius: v[2] as f64,
        })
        .collect(),
    )
  }
}
impl_passthrough_task!(HoughCirclesTask, Vec<Circle>);

// ─── HoughLinesTask ──────────────────────────────────────────────────────────

#[napi(object)]
#[derive(Clone, Debug)]
pub struct HoughLine {
  pub rho: f64,
  pub theta: f64,
}

pub(crate) struct HoughLinesTask {
  src: Arc<Mat>,
  rho: f64,
  theta: f64,
  threshold: i32,
}

impl HoughLinesTask {
  fn do_compute(&mut self) -> Result<Vec<HoughLine>> {
    let mut lines = Mat::default();
    cv_err!(imgproc::hough_lines(
      &*self.src,
      &mut lines,
      self.rho,
      self.theta,
      self.threshold,
      0.0,
      0.0,
      0.0,
      std::f64::consts::PI,
    ))?;
    // hough_lines returns Nx1 CV_32FC2 (rho, theta per element).
    // When no lines are found, OpenCV may leave the output as a default empty
    // Mat (CV_8UC1), so guard before calling data_typed to avoid a type mismatch error.
    if lines.empty() {
      return Ok(Vec::new());
    }
    let data = cv_err!(lines.data_typed::<opencv::core::Vec2f>())?;
    Ok(
      data
        .iter()
        .map(|v| HoughLine {
          rho: v[0] as f64,
          theta: v[1] as f64,
        })
        .collect(),
    )
  }
}
impl_passthrough_task!(HoughLinesTask, Vec<HoughLine>);

// ─── getStructuringElement (sync) ────────────────────────────────────────────

#[napi(js_name = "getStructuringElement")]
pub fn get_structuring_element(
  shape: MorphShape,
  ksize_width: i32,
  ksize_height: i32,
) -> Result<JSMat> {
  let kernel = cv_err!(imgproc::get_structuring_element(
    shape as i32,
    opencv::core::Size::new(ksize_width, ksize_height),
    opencv::core::Point::new(-1, -1),
  ))?;
  Ok(JSMat {
    mat: Arc::new(kernel),
  })
}

// ─── arcLength (sync) ────────────────────────────────────────────────────────

#[napi(js_name = "arcLength")]
pub fn arc_length(contour: Vec<Point>, closed: bool) -> Result<f64> {
  let pts = pts_to_cv(&contour);
  cv_err!(imgproc::arc_length(&pts, closed))
}

// ─── approxPolyDP (sync) ─────────────────────────────────────────────────────

#[napi(js_name = "approxPolyDP")]
pub fn approx_poly_dp(contour: Vec<Point>, epsilon: f64, closed: bool) -> Result<Vec<Point>> {
  let pts = pts_to_cv(&contour);
  let mut approx: opencv::core::Vector<opencv::core::Point> = opencv::core::Vector::new();
  cv_err!(imgproc::approx_poly_dp(&pts, &mut approx, epsilon, closed))?;
  Ok(approx.into_iter().map(Into::into).collect())
}

// ─── convexHull (sync) ───────────────────────────────────────────────────────

#[napi(js_name = "convexHull")]
pub fn convex_hull(points: Vec<Point>) -> Result<Vec<Point>> {
  let pts = pts_to_cv(&points);
  let mut hull: opencv::core::Vector<opencv::core::Point> = opencv::core::Vector::new();
  cv_err!(imgproc::convex_hull(&pts, &mut hull, false, true))?;
  Ok(hull.into_iter().map(Into::into).collect())
}

// ─── MomentsResult / MomentsTask ─────────────────────────────────────────────

#[napi(object)]
pub struct MomentsResult {
  pub m00: f64,
  pub m10: f64,
  pub m01: f64,
  pub m20: f64,
  pub m11: f64,
  pub m02: f64,
  pub m30: f64,
  pub m21: f64,
  pub m12: f64,
  pub m03: f64,
  pub mu20: f64,
  pub mu11: f64,
  pub mu02: f64,
  pub mu30: f64,
  pub mu21: f64,
  pub mu12: f64,
  pub mu03: f64,
  pub nu20: f64,
  pub nu11: f64,
  pub nu02: f64,
  pub nu30: f64,
  pub nu21: f64,
  pub nu12: f64,
  pub nu03: f64,
}

pub(crate) struct MomentsTask {
  src: Arc<Mat>,
  binary_image: bool,
}

impl MomentsTask {
  fn do_compute(&mut self) -> Result<MomentsResult> {
    let m = cv_err!(imgproc::moments(&*self.src, self.binary_image))?;
    Ok(MomentsResult {
      m00: m.m00,
      m10: m.m10,
      m01: m.m01,
      m20: m.m20,
      m11: m.m11,
      m02: m.m02,
      m30: m.m30,
      m21: m.m21,
      m12: m.m12,
      m03: m.m03,
      mu20: m.mu20,
      mu11: m.mu11,
      mu02: m.mu02,
      mu30: m.mu30,
      mu21: m.mu21,
      mu12: m.mu12,
      mu03: m.mu03,
      nu20: m.nu20,
      nu11: m.nu11,
      nu02: m.nu02,
      nu30: m.nu30,
      nu21: m.nu21,
      nu12: m.nu12,
      nu03: m.nu03,
    })
  }
}
impl_passthrough_task!(MomentsTask, MomentsResult);

// ─── AdaptiveThresholdTask ───────────────────────────────────────────────────

pub(crate) struct AdaptiveThresholdTask {
  src: Arc<Mat>,
  max_value: f64,
  adaptive_method: i32,
  threshold_type: i32,
  block_size: i32,
  c: f64,
}

impl AdaptiveThresholdTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(imgproc::adaptive_threshold(
      &*self.src,
      &mut dst,
      self.max_value,
      self.adaptive_method,
      self.threshold_type,
      self.block_size,
      self.c,
    ))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(AdaptiveThresholdTask);

// ─── HoughLinesPTask ─────────────────────────────────────────────────────────

#[napi(object)]
#[derive(Clone)]
pub struct LineSegment {
  pub x1: i32,
  pub y1: i32,
  pub x2: i32,
  pub y2: i32,
}

pub(crate) struct HoughLinesPTask {
  src: Arc<Mat>,
  rho: f64,
  theta: f64,
  threshold: i32,
  min_line_length: f64,
  max_line_gap: f64,
}

impl HoughLinesPTask {
  fn do_compute(&mut self) -> Result<Vec<LineSegment>> {
    let mut lines = Mat::default();
    cv_err!(imgproc::hough_lines_p(
      &*self.src,
      &mut lines,
      self.rho,
      self.theta,
      self.threshold,
      self.min_line_length,
      self.max_line_gap,
    ))?;
    // hough_lines_p returns Nx1 CV_32SC4 (x1,y1,x2,y2 per element).
    if lines.empty() {
      return Ok(Vec::new());
    }
    let data = cv_err!(lines.data_typed::<opencv::core::Vec4i>())?;
    Ok(
      data
        .iter()
        .map(|seg| LineSegment {
          x1: seg[0],
          y1: seg[1],
          x2: seg[2],
          y2: seg[3],
        })
        .collect(),
    )
  }
}
impl_passthrough_task!(HoughLinesPTask, Vec<LineSegment>);

// ─── GetPerspectiveTransformTask ─────────────────────────────────────────────

pub(crate) struct GetPerspectiveTransformTask {
  src_pts: Vec<PointF64>,
  dst_pts: Vec<PointF64>,
}

impl GetPerspectiveTransformTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let src = to_point2f::<4>(&self.src_pts)?;
    let dst = to_point2f::<4>(&self.dst_pts)?;
    let m = cv_err!(imgproc::get_perspective_transform_slice_def(&src, &dst))?;
    Ok(JSMat { mat: Arc::new(m) })
  }
}
impl_mat_task!(GetPerspectiveTransformTask);

fn to_point2f<const N: usize>(pts: &[PointF64]) -> napi::Result<[opencv::core::Point2f; N]> {
  if pts.len() != N {
    return Err(napi::Error::new(
      napi::Status::InvalidArg,
      format!("Expected exactly {N} points"),
    ));
  }
  Ok(std::array::from_fn(|i| pts[i].into()))
}

#[napi(js_name = "getPerspectiveTransform")]
pub fn get_perspective_transform(
  src_pts: Vec<PointF64>,
  dst_pts: Vec<PointF64>,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<GetPerspectiveTransformTask> {
  AsyncTask::with_optional_signal(
    GetPerspectiveTransformTask { src_pts, dst_pts },
    abort_signal,
  )
}

// ─── GetAffineTransformTask ──────────────────────────────────────────────────

pub(crate) struct GetAffineTransformTask {
  src_pts: Vec<PointF64>,
  dst_pts: Vec<PointF64>,
}

impl GetAffineTransformTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let src = to_point2f::<3>(&self.src_pts)?;
    let dst = to_point2f::<3>(&self.dst_pts)?;
    let m = cv_err!(imgproc::get_affine_transform_slice(&src, &dst))?;
    Ok(JSMat { mat: Arc::new(m) })
  }
}
impl_mat_task!(GetAffineTransformTask);

#[napi(js_name = "getAffineTransform")]
pub fn get_affine_transform(
  src_pts: Vec<PointF64>,
  dst_pts: Vec<PointF64>,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<GetAffineTransformTask> {
  AsyncTask::with_optional_signal(GetAffineTransformTask { src_pts, dst_pts }, abort_signal)
}

// ─── JSMat methods (feature-detection operations) ────────────────────────────

#[napi]
impl JSMat {
  #[napi(js_name = "findContours")]
  pub fn find_contours(
    &self,
    mode: ContourRetrievalMode,
    method: ContourApproximation,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<FindContoursTask> {
    AsyncTask::with_optional_signal(
      FindContoursTask {
        src: self.mat.clone(),
        mode: mode as i32,
        method: method as i32,
      },
      abort_signal,
    )
  }

  #[allow(clippy::too_many_arguments)]
  #[napi(js_name = "goodFeaturesToTrack")]
  pub fn good_features_to_track(
    &self,
    max_corners: i32,
    quality_level: f64,
    min_distance: f64,
    block_size: Option<i32>,
    use_harris: Option<bool>,
    k: Option<f64>,
    mask: Option<&JSMat>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<GoodFeaturesToTrackTask> {
    AsyncTask::with_optional_signal(
      GoodFeaturesToTrackTask {
        src: self.mat.clone(),
        max_corners,
        quality_level,
        min_distance,
        block_size: block_size.unwrap_or(3),
        use_harris: use_harris.unwrap_or(false),
        k: k.unwrap_or(0.04),
        mask: mask.map(|m| m.mat.clone()),
      },
      abort_signal,
    )
  }

  #[allow(clippy::too_many_arguments)]
  #[napi(js_name = "houghCircles")]
  pub fn hough_circles(
    &self,
    dp: f64,
    min_dist: f64,
    param1: Option<f64>,
    param2: Option<f64>,
    min_radius: Option<i32>,
    max_radius: Option<i32>,
    method: Option<HoughMethod>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<HoughCirclesTask> {
    AsyncTask::with_optional_signal(
      HoughCirclesTask {
        src: self.mat.clone(),
        method: method.map(|m| m as i32).unwrap_or(imgproc::HOUGH_GRADIENT),
        dp,
        min_dist,
        param1: param1.unwrap_or(100.0),
        param2: param2.unwrap_or(100.0),
        min_radius: min_radius.unwrap_or(0),
        max_radius: max_radius.unwrap_or(0),
      },
      abort_signal,
    )
  }

  #[napi(js_name = "houghLines")]
  pub fn hough_lines(
    &self,
    rho: f64,
    theta: f64,
    threshold: i32,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<HoughLinesTask> {
    AsyncTask::with_optional_signal(
      HoughLinesTask {
        src: self.mat.clone(),
        rho,
        theta,
        threshold,
      },
      abort_signal,
    )
  }

  #[napi(js_name = "houghLinesP")]
  pub fn hough_lines_p(
    &self,
    rho: f64,
    theta: f64,
    threshold: i32,
    min_line_length: Option<f64>,
    max_line_gap: Option<f64>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<HoughLinesPTask> {
    AsyncTask::with_optional_signal(
      HoughLinesPTask {
        src: self.mat.clone(),
        rho,
        theta,
        threshold,
        min_line_length: min_line_length.unwrap_or(0.0),
        max_line_gap: max_line_gap.unwrap_or(0.0),
      },
      abort_signal,
    )
  }

  #[napi(js_name = "adaptiveThreshold")]
  pub fn adaptive_threshold(
    &self,
    max_value: f64,
    adaptive_method: AdaptiveThresholdType,
    threshold_type: ThresholdType,
    block_size: i32,
    c: f64,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<AdaptiveThresholdTask> {
    AsyncTask::with_optional_signal(
      AdaptiveThresholdTask {
        src: self.mat.clone(),
        max_value,
        adaptive_method: adaptive_method as i32,
        threshold_type: threshold_type as i32,
        block_size,
        c,
      },
      abort_signal,
    )
  }

  #[napi(js_name = "moments")]
  pub fn moments_fn(
    &self,
    binary_image: Option<bool>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<MomentsTask> {
    AsyncTask::with_optional_signal(
      MomentsTask {
        src: self.mat.clone(),
        binary_image: binary_image.unwrap_or(false),
      },
      abort_signal,
    )
  }
}
