use crate::{JSMat, OpenCVError};
use napi::bindgen_prelude::*;
use opencv::core as cv_core;

/// Flip task
pub struct FlipTask {
  source_mat: cv_core::Mat,
  flip_code: i32,
}

#[napi]
impl Task for FlipTask {
  type Output = JSMat;
  type JsValue = JSMat;

  fn compute(&mut self) -> Result<Self::Output> {
    let mut dst = cv_core::Mat::default();
    cv_core::flip(&self.source_mat, &mut dst, self.flip_code).map_err(OpenCVError)?;
    Ok(JSMat { mat: dst })
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

/// Rotate task
pub struct RotateTask {
  source_mat: cv_core::Mat,
  rotate_code: i32,
}

#[napi]
impl Task for RotateTask {
  type Output = JSMat;
  type JsValue = JSMat;

  fn compute(&mut self) -> Result<Self::Output> {
    let mut dst = cv_core::Mat::default();
    cv_core::rotate(&self.source_mat, &mut dst, self.rotate_code).map_err(OpenCVError)?;
    Ok(JSMat { mat: dst })
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

/// Merge task
pub struct MergeTask {
  mats: Vec<cv_core::Mat>,
}

#[napi]
impl Task for MergeTask {
  type Output = JSMat;
  type JsValue = JSMat;

  fn compute(&mut self) -> Result<Self::Output> {
    let mut dst = cv_core::Mat::default();
    let vec: cv_core::Vector<cv_core::Mat> = cv_core::Vector::from_iter(self.mats.clone());
    cv_core::merge(&vec, &mut dst).map_err(OpenCVError)?;
    Ok(JSMat { mat: dst })
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

/// Split task
pub struct SplitTask {
  source_mat: cv_core::Mat,
}

#[napi]
impl Task for SplitTask {
  type Output = Vec<JSMat>;
  type JsValue = Vec<JSMat>;

  fn compute(&mut self) -> Result<Self::Output> {
    let mut vec = cv_core::Vector::new();
    cv_core::split(&self.source_mat, &mut vec).map_err(OpenCVError)?;
    let result: Vec<JSMat> = vec.iter().map(|mat| JSMat { mat }).collect();
    Ok(result)
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

/// InRange task
pub struct InRangeTask {
  source_mat: cv_core::Mat,
  lower_bound: cv_core::Scalar,
  upper_bound: cv_core::Scalar,
}

#[napi]
impl Task for InRangeTask {
  type Output = JSMat;
  type JsValue = JSMat;

  fn compute(&mut self) -> Result<Self::Output> {
    let mut dst = cv_core::Mat::default();
    cv_core::in_range(
      &self.source_mat,
      &self.lower_bound,
      &self.upper_bound,
      &mut dst,
    )
    .map_err(OpenCVError)?;
    Ok(JSMat { mat: dst })
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

/// Flip an image horizontally, vertically, or both.
/// 
/// # Arguments
/// * `src` - Source image
/// * `flip_code` - A flag to specify how to flip the image:
///   - 0: flip vertically
///   - positive value (e.g., 1): flip horizontally
///   - negative value (e.g., -1): flip both horizontally and vertically
#[napi]
pub fn flip(
  #[napi(ts_arg_type = "JSMat")] src: &JSMat,
  flip_code: i32,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<FlipTask> {
  AsyncTask::with_optional_signal(
    FlipTask {
      source_mat: src.mat.clone(),
      flip_code,
    },
    abort_signal,
  )
}

/// Rotate an image by 90, 180, or 270 degrees.
/// 
/// # Arguments
/// * `src` - Source image
/// * `rotate_code` - An enum to specify how to rotate the array:
///   - ROTATE_90_CLOCKWISE: Rotate 90 degrees clockwise
///   - ROTATE_180: Rotate 180 degrees
///   - ROTATE_90_COUNTERCLOCKWISE: Rotate 270 degrees clockwise (90 degrees counterclockwise)
#[napi]
pub fn rotate(
  #[napi(ts_arg_type = "JSMat")] src: &JSMat,
  rotate_code: i32,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<RotateTask> {
  AsyncTask::with_optional_signal(
    RotateTask {
      source_mat: src.mat.clone(),
      rotate_code,
    },
    abort_signal,
  )
}

/// Merge several single-channel arrays into a multi-channel array.
/// 
/// # Arguments
/// * `mats` - Array of single-channel matrices to be merged
#[napi]
pub fn merge(
  #[napi(ts_arg_type = "JSMat[]")] mats: Vec<&JSMat>,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<MergeTask> {
  let mat_vec: Vec<cv_core::Mat> = mats.iter().map(|m| m.mat.clone()).collect();
  AsyncTask::with_optional_signal(MergeTask { mats: mat_vec }, abort_signal)
}

/// Split a multi-channel array into several single-channel arrays.
/// 
/// # Arguments
/// * `src` - Source multi-channel array
#[napi]
pub fn split(
  #[napi(ts_arg_type = "JSMat")] src: &JSMat,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<SplitTask> {
  AsyncTask::with_optional_signal(
    SplitTask {
      source_mat: src.mat.clone(),
    },
    abort_signal,
  )
}

/// Check if array elements lie between the elements of two other arrays.
/// 
/// # Arguments
/// * `src` - Source array
/// * `lower_bound` - Array of lower bounds (format: [b, g, r, a])
/// * `upper_bound` - Array of upper bounds (format: [b, g, r, a])
#[napi]
pub fn in_range(
  #[napi(ts_arg_type = "JSMat")] src: &JSMat,
  lower_bound: Vec<f64>,
  upper_bound: Vec<f64>,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<InRangeTask> {
  let lower = match lower_bound.len() {
    1 => cv_core::Scalar::new(lower_bound[0], 0.0, 0.0, 0.0),
    2 => cv_core::Scalar::new(lower_bound[0], lower_bound[1], 0.0, 0.0),
    3 => cv_core::Scalar::new(lower_bound[0], lower_bound[1], lower_bound[2], 0.0),
    4 => cv_core::Scalar::new(
      lower_bound[0],
      lower_bound[1],
      lower_bound[2],
      lower_bound[3],
    ),
    _ => cv_core::Scalar::default(),
  };

  let upper = match upper_bound.len() {
    1 => cv_core::Scalar::new(upper_bound[0], 0.0, 0.0, 0.0),
    2 => cv_core::Scalar::new(upper_bound[0], upper_bound[1], 0.0, 0.0),
    3 => cv_core::Scalar::new(upper_bound[0], upper_bound[1], upper_bound[2], 0.0),
    4 => cv_core::Scalar::new(
      upper_bound[0],
      upper_bound[1],
      upper_bound[2],
      upper_bound[3],
    ),
    _ => cv_core::Scalar::default(),
  };

  AsyncTask::with_optional_signal(
    InRangeTask {
      source_mat: src.mat.clone(),
      lower_bound: lower,
      upper_bound: upper,
    },
    abort_signal,
  )
}
