use crate::constants::{
  BorderType, ColorCode, FlipCode, InterpolationFlag, MorphType, NormType, TemplateMatchMode,
  ThresholdType,
};
use crate::cv_err;
use crate::impl_mat_task;
use crate::impl_passthrough_task;
use crate::types::{MinMaxResult, Rect, Size};
use crate::utils::vec4;
use napi::bindgen_prelude::*;
use opencv::{
  core::{Mat, Scalar, ToInputArray, Vector},
  imgproc::match_template,
  prelude::*,
};
use std::sync::Arc;

#[napi(js_name = "Mat")]
pub struct JSMat {
  pub(crate) mat: Arc<Mat>,
}

// ─── AsyncTask: MatchTemplate ────────────────────────────────────────────────

struct ConvertToTask {
  src: Arc<Mat>,
  rtype: i32,
  alpha: f64,
  beta: f64,
}
impl ConvertToTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(self
      .src
      .convert_to(&mut dst, self.rtype, self.alpha, self.beta))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(ConvertToTask);

struct MatchTemplateTask {
  src: Arc<Mat>,
  tpl: Arc<Mat>,
  method: i32,
}

impl MatchTemplateTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut result = Mat::default();
    cv_err!(match_template(
      &*self.src,
      &*self.tpl,
      &mut result,
      self.method,
      &opencv::core::no_array()
    ))?;
    Ok(JSMat {
      mat: Arc::new(result),
    })
  }
}
impl_mat_task!(MatchTemplateTask);

struct MinMaxLocTask {
  src: Arc<Mat>,
}

impl MinMaxLocTask {
  fn do_compute(&mut self) -> Result<MinMaxResult> {
    let mut min_val = 0f64;
    let mut max_val = 0f64;
    let mut min_loc = opencv::core::Point::new(0, 0);
    let mut max_loc = opencv::core::Point::new(0, 0);
    cv_err!(opencv::core::min_max_loc(
      &*self.src,
      Some(&mut min_val),
      Some(&mut max_val),
      Some(&mut min_loc),
      Some(&mut max_loc),
      &opencv::core::no_array()
    ))?;
    Ok(MinMaxResult {
      min_val,
      max_val,
      min_loc: min_loc.into(),
      max_loc: max_loc.into(),
    })
  }
}
impl_passthrough_task!(MinMaxLocTask, MinMaxResult);

struct ThresholdTask {
  src: Arc<Mat>,
  thresh: f64,
  maxval: f64,
  typ: i32,
}

impl ThresholdTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(opencv::imgproc::threshold(
      &*self.src,
      &mut dst,
      self.thresh,
      self.maxval,
      self.typ
    ))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(ThresholdTask);

struct MatchTemplateAllTask {
  src: Arc<Mat>,
  tpl: Arc<Mat>,
  method: i32,
  score: f64,
  nms_threshold: f64,
}

impl MatchTemplateAllTask {
  fn do_compute(&mut self) -> Result<Vec<Rect>> {
    let mut result = Mat::default();
    cv_err!(match_template(
      &*self.src,
      &*self.tpl,
      &mut result,
      self.method,
      &opencv::core::no_array()
    ))?;
    // match_template always outputs CV_32F; data_typed enforces that at runtime
    // and requires continuity — make continuous first as a safeguard.
    if !result.is_continuous() {
      let mut cont = Mat::default();
      cv_err!(result.copy_to(&mut cont))?;
      result = cont;
    }
    let cols = result.cols() as usize;
    let result_data = cv_err!(result.data_typed::<f32>())?;
    let score_f32 = self.score as f32;
    let mut matches = Vec::new();
    let mut scores = Vec::new();
    for (idx, &value) in result_data.iter().enumerate() {
      if value >= score_f32 {
        matches.push(Rect {
          x: (idx % cols) as i32,
          y: (idx / cols) as i32,
          width: self.tpl.cols(),
          height: self.tpl.rows(),
        });
        scores.push(value);
      }
    }
    if matches.is_empty() {
      return Ok(Vec::new());
    }
    let indices = crate::dnn::apply_nms(&matches, &scores, score_f32, self.nms_threshold as f32)?;
    indices
      .iter()
      .map(|i| {
        let idx = usize::try_from(i).map_err(|_| {
          Error::new(
            Status::GenericFailure,
            format!("NMS returned negative index {i}"),
          )
        })?;
        matches.get(idx).copied().ok_or_else(|| {
          Error::new(
            Status::GenericFailure,
            format!("NMS returned out-of-range index {i}"),
          )
        })
      })
      .collect::<Result<Vec<_>>>()
  }
}
impl_passthrough_task!(MatchTemplateAllTask, Vec<Rect>);

struct CvtColorTask {
  src: Arc<Mat>,
  code: i32,
}
impl CvtColorTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(opencv::imgproc::cvt_color(
      &*self.src, &mut dst, self.code, 0
    ))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(CvtColorTask);

struct ResizeTask {
  src: Arc<Mat>,
  width: i32,
  height: i32,
  fx: f64,
  fy: f64,
  interpolation: i32, // stored as i32 for OpenCV FFI
}
impl ResizeTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(opencv::imgproc::resize(
      &*self.src,
      &mut dst,
      opencv::core::Size::new(self.width, self.height),
      self.fx,
      self.fy,
      self.interpolation
    ))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(ResizeTask);

struct GaussianBlurTask {
  src: Arc<Mat>,
  ksize_width: i32,
  ksize_height: i32,
  sigma_x: f64,
  sigma_y: f64,
}
impl GaussianBlurTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(opencv::imgproc::gaussian_blur(
      &*self.src,
      &mut dst,
      opencv::core::Size::new(self.ksize_width, self.ksize_height),
      self.sigma_x,
      self.sigma_y,
      opencv::core::BORDER_DEFAULT
    ))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(GaussianBlurTask);

struct CannyTask {
  src: Arc<Mat>,
  threshold1: f64,
  threshold2: f64,
  aperture_size: i32,
  l2gradient: bool,
}
impl CannyTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(opencv::imgproc::canny(
      &*self.src,
      &mut dst,
      self.threshold1,
      self.threshold2,
      self.aperture_size,
      self.l2gradient
    ))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(CannyTask);

macro_rules! impl_morph_task {
  ($task:ident, $fn:path) => {
    pub(crate) struct $task {
      src: Arc<Mat>,
      kernel: Arc<Mat>,
      iterations: i32,
    }
    impl $task {
      fn do_compute(&mut self) -> Result<JSMat> {
        let mut dst = Mat::default();
        cv_err!($fn(
          &*self.src,
          &mut dst,
          &*self.kernel,
          opencv::core::Point::new(-1, -1),
          self.iterations,
          opencv::core::BORDER_CONSTANT,
          Scalar::default()
        ))?;
        Ok(JSMat { mat: Arc::new(dst) })
      }
    }
    impl_mat_task!($task);
  };
}

impl_morph_task!(DilateTask, opencv::imgproc::dilate);
impl_morph_task!(ErodeTask, opencv::imgproc::erode);

struct MorphologyExTask {
  src: Arc<Mat>,
  op: i32,
  kernel: Arc<Mat>,
  iterations: i32,
}
impl MorphologyExTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(opencv::imgproc::morphology_ex(
      &*self.src,
      &mut dst,
      self.op,
      &*self.kernel,
      opencv::core::Point::new(-1, -1),
      self.iterations,
      opencv::core::BORDER_CONSTANT,
      Scalar::default()
    ))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(MorphologyExTask);

macro_rules! impl_warp_task {
  ($task:ident, $fn:path) => {
    pub(crate) struct $task {
      src: Arc<Mat>,
      m: Arc<Mat>,
      dst_width: i32,
      dst_height: i32,
      flags: i32,
      border_mode: i32,
    }
    impl $task {
      fn do_compute(&mut self) -> Result<JSMat> {
        let mut dst = Mat::default();
        cv_err!($fn(
          &*self.src,
          &mut dst,
          &*self.m,
          opencv::core::Size::new(self.dst_width, self.dst_height),
          self.flags,
          self.border_mode,
          Scalar::default()
        ))?;
        Ok(JSMat { mat: Arc::new(dst) })
      }
    }
    impl_mat_task!($task);
  };
}

impl_warp_task!(WarpAffineTask, opencv::imgproc::warp_affine);

struct FlipTask {
  src: Arc<Mat>,
  flip_code: i32,
}
impl FlipTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(opencv::core::flip(&*self.src, &mut dst, self.flip_code))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(FlipTask);

struct CropTask {
  src: Arc<Mat>,
  x: i32,
  y: i32,
  width: i32,
  height: i32,
}
impl CropTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let roi = opencv::core::Rect::new(self.x, self.y, self.width, self.height);
    let mut dst = Mat::default();
    {
      let cropped = cv_err!(Mat::roi(&*self.src, roi))?;
      cv_err!(cropped.copy_to(&mut dst))?;
    }
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(CropTask);

struct InRangeTask {
  src: Arc<Mat>,
  lower: crate::types::Scalar,
  upper: crate::types::Scalar,
}
impl InRangeTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(opencv::core::in_range(
      &*self.src,
      &opencv::core::Scalar::from(self.lower),
      &opencv::core::Scalar::from(self.upper),
      &mut dst,
    ))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(InRangeTask);

macro_rules! impl_bitwise_binary_task {
  ($task:ident, $fn:path) => {
    pub(crate) struct $task {
      src1: Arc<Mat>,
      src2: Arc<Mat>,
      mask: Option<Arc<Mat>>,
    }
    impl $task {
      fn do_compute(&mut self) -> Result<JSMat> {
        let mut dst = Mat::default();
        // Unify optional mask: both Mat::input_array() and no_array().input_array()
        // yield the same BoxedRef<'_, _InputArray> type, eliminating the duplicate
        // call sites that previously duplicated every argument for each branch.
        let no_arr = opencv::core::no_array();
        let mask_ia = match &self.mask {
          Some(m) => cv_err!(m.as_ref().input_array())?,
          None => cv_err!(no_arr.input_array())?,
        };
        cv_err!($fn(&*self.src1, &*self.src2, &mut dst, &mask_ia))?;
        Ok(JSMat { mat: Arc::new(dst) })
      }
    }
    impl_mat_task!($task);
  };
}

impl_bitwise_binary_task!(BitwiseAndTask, opencv::core::bitwise_and);
impl_bitwise_binary_task!(BitwiseOrTask, opencv::core::bitwise_or);
impl_bitwise_binary_task!(BitwiseXorTask, opencv::core::bitwise_xor);

struct BitwiseNotTask {
  src: Arc<Mat>,
  mask: Option<Arc<Mat>>,
}
impl BitwiseNotTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    let no_arr = opencv::core::no_array();
    let mask_ia = match &self.mask {
      Some(m) => cv_err!(m.as_ref().input_array())?,
      None => cv_err!(no_arr.input_array())?,
    };
    cv_err!(opencv::core::bitwise_not(&*self.src, &mut dst, &mask_ia))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(BitwiseNotTask);

struct SplitTask {
  src: Arc<Mat>,
}
impl SplitTask {
  fn do_compute(&mut self) -> Result<Vec<JSMat>> {
    let n_channels = self.src.channels() as usize;
    let mut channels: Vector<Mat> = Vector::with_capacity(n_channels);
    cv_err!(opencv::core::split(&*self.src, &mut channels))?;
    Ok(
      channels
        .into_iter()
        .map(|mat| JSMat { mat: Arc::new(mat) })
        .collect(),
    )
  }
}
impl_passthrough_task!(SplitTask, Vec<JSMat>);

// Returns a non-owning Mat view over `m`'s pixel data for continuous matrices,
// or a deep clone for non-continuous ones.
//
// Safety contract: the caller must keep the source Arc<Mat> alive for the
// entire duration the returned Mat is in use. OpenCV read-only operations
// (merge, hconcat, vconcat) satisfy this when the source Arc lives in the
// calling Task's `self`.
fn mat_view_or_clone(m: &Mat) -> napi::Result<Mat> {
  if m.is_continuous() {
    // Safety: see contract above.
    let ptr = cv_err!(m.data_bytes())?.as_ptr();
    let view = unsafe {
      cv_err!(Mat::new_rows_cols_with_data_unsafe_def(
        m.rows(),
        m.cols(),
        m.typ(),
        ptr as *mut std::ffi::c_void,
      ))?
    };
    Ok(view)
  } else {
    // Non-continuous Mat: clone to get contiguous memory required by
    // merge / hconcat / vconcat.
    cv_err!(m.try_clone())
  }
}

struct MergeTask {
  channels: Vec<Arc<Mat>>,
}
impl MergeTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut channels = Vector::<Mat>::new();
    for m in &self.channels {
      channels.push(mat_view_or_clone(m)?);
    }
    let mut dst = Mat::default();
    cv_err!(opencv::core::merge(&channels, &mut dst))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(MergeTask);

struct NormalizeTask {
  src: Arc<Mat>,
  alpha: f64,
  beta: f64,
  norm_type: i32,
  dtype: i32,
}
impl NormalizeTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(opencv::core::normalize(
      &*self.src,
      &mut dst,
      self.alpha,
      self.beta,
      self.norm_type,
      self.dtype,
      &opencv::core::no_array()
    ))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(NormalizeTask);

struct AddWeightedTask {
  src1: Arc<Mat>,
  alpha: f64,
  src2: Arc<Mat>,
  beta: f64,
  gamma: f64,
  dtype: i32,
}
impl AddWeightedTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(opencv::core::add_weighted(
      &*self.src1,
      self.alpha,
      &*self.src2,
      self.beta,
      self.gamma,
      &mut dst,
      self.dtype
    ))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(AddWeightedTask);

struct MedianBlurTask {
  src: Arc<Mat>,
  ksize: i32,
}
impl MedianBlurTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(opencv::imgproc::median_blur(
      &*self.src, &mut dst, self.ksize
    ))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(MedianBlurTask);

struct BilateralFilterTask {
  src: Arc<Mat>,
  d: i32,
  sigma_color: f64,
  sigma_space: f64,
}
impl BilateralFilterTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(opencv::imgproc::bilateral_filter(
      &*self.src,
      &mut dst,
      self.d,
      self.sigma_color,
      self.sigma_space,
      opencv::core::BORDER_DEFAULT
    ))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(BilateralFilterTask);

struct SobelTask {
  src: Arc<Mat>,
  ddepth: i32,
  dx: i32,
  dy: i32,
  ksize: i32,
  scale: f64,
  delta: f64,
}
impl SobelTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(opencv::imgproc::sobel(
      &*self.src,
      &mut dst,
      self.ddepth,
      self.dx,
      self.dy,
      self.ksize,
      self.scale,
      self.delta,
      opencv::core::BORDER_DEFAULT
    ))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(SobelTask);

struct LaplacianTask {
  src: Arc<Mat>,
  ddepth: i32,
  ksize: i32,
  scale: f64,
  delta: f64,
}
impl LaplacianTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(opencv::imgproc::laplacian(
      &*self.src,
      &mut dst,
      self.ddepth,
      self.ksize,
      self.scale,
      self.delta,
      opencv::core::BORDER_DEFAULT
    ))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(LaplacianTask);

struct EqualizeHistTask {
  src: Arc<Mat>,
}
impl EqualizeHistTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(opencv::imgproc::equalize_hist(&*self.src, &mut dst))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(EqualizeHistTask);

struct CopyMakeBorderTask {
  src: Arc<Mat>,
  top: i32,
  bottom: i32,
  left: i32,
  right: i32,
  border_type: i32,
  value: [f64; 4],
}
impl CopyMakeBorderTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    let val = Scalar::new(self.value[0], self.value[1], self.value[2], self.value[3]);
    cv_err!(opencv::core::copy_make_border(
      &*self.src,
      &mut dst,
      self.top,
      self.bottom,
      self.left,
      self.right,
      self.border_type,
      val
    ))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(CopyMakeBorderTask);

pub(crate) struct Filter2DTask {
  src: Arc<Mat>,
  ddepth: i32,
  kernel: Arc<Mat>,
  delta: f64,
}
impl Filter2DTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(opencv::imgproc::filter_2d(
      &*self.src,
      &mut dst,
      self.ddepth,
      &*self.kernel,
      opencv::core::Point::new(-1, -1),
      self.delta,
      opencv::core::BORDER_DEFAULT
    ))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(Filter2DTask);

struct AbsDiffTask {
  src1: Arc<Mat>,
  src2: Arc<Mat>,
}
impl AbsDiffTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(opencv::core::absdiff(&*self.src1, &*self.src2, &mut dst))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(AbsDiffTask);

impl_warp_task!(WarpPerspectiveTask, opencv::imgproc::warp_perspective);

struct ConcatTask {
  mats: Vec<Arc<Mat>>,
  horizontal: bool,
}
impl ConcatTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut src = Vector::<Mat>::new();
    for m in &self.mats {
      src.push(mat_view_or_clone(m)?);
    }
    let mut dst = Mat::default();
    if self.horizontal {
      cv_err!(opencv::core::hconcat(&src, &mut dst))?;
    } else {
      cv_err!(opencv::core::vconcat(&src, &mut dst))?;
    }
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(ConcatTask);

struct CountNonZeroTask {
  src: Arc<Mat>,
}
impl CountNonZeroTask {
  fn do_compute(&mut self) -> Result<i32> {
    cv_err!(opencv::core::count_non_zero(&*self.src))
  }
}
impl_passthrough_task!(CountNonZeroTask, i32);

struct MeanTask {
  src: Arc<Mat>,
}
impl MeanTask {
  fn do_compute(&mut self) -> Result<Vec<f64>> {
    let s = cv_err!(opencv::core::mean_def(&*self.src))?;
    Ok(vec![s[0], s[1], s[2], s[3]])
  }
}
impl_passthrough_task!(MeanTask, Vec<f64>);

struct TransposeTask {
  src: Arc<Mat>,
}
impl TransposeTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(opencv::core::transpose(&*self.src, &mut dst))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(TransposeTask);

macro_rules! impl_two_src_task {
  ($task:ident, $fn:path) => {
    pub(crate) struct $task {
      src1: Arc<Mat>,
      src2: Arc<Mat>,
    }
    impl $task {
      fn do_compute(&mut self) -> Result<JSMat> {
        let mut dst = Mat::default();
        // Use no_array() instead of Mat::default() — no_array() is the canonical
        // OpenCV sentinel for "no mask", avoids constructing a Mat object.
        cv_err!($fn(
          &*self.src1,
          &*self.src2,
          &mut dst,
          &opencv::core::no_array(),
          -1,
        ))?;
        Ok(JSMat { mat: Arc::new(dst) })
      }
    }
    impl_mat_task!($task);
  };
}

impl_two_src_task!(AddTask, opencv::core::add);
impl_two_src_task!(SubtractTask, opencv::core::subtract);

struct MultiplyTask {
  src1: Arc<Mat>,
  src2: Arc<Mat>,
  scale: f64,
}
impl MultiplyTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut dst = Mat::default();
    cv_err!(opencv::core::multiply(
      &*self.src1,
      &*self.src2,
      &mut dst,
      self.scale,
      -1
    ))?;
    Ok(JSMat { mat: Arc::new(dst) })
  }
}
impl_mat_task!(MultiplyTask);

impl JSMat {
  pub(crate) fn get_mut(&mut self) -> napi::Result<&mut Mat> {
    Arc::get_mut(&mut self.mat).ok_or_else(|| {
      napi::Error::from_reason(
        "Mat is shared; cannot mutate in place — ensure no async task holds a reference",
      )
    })
  }
}

impl Default for JSMat {
  fn default() -> Self {
    Self {
      mat: Arc::new(Mat::default()),
    }
  }
}

#[napi]
impl JSMat {
  #[napi(constructor)]
  pub fn new() -> Self {
    Self::default()
  }

  #[napi(factory)]
  pub fn zeros(rows: i32, cols: i32, mat_type: i32) -> Result<JSMat> {
    let expr = cv_err!(Mat::zeros(rows, cols, mat_type))?;
    let mat = cv_err!(expr.to_mat())?;
    Ok(JSMat { mat: Arc::new(mat) })
  }

  #[napi(factory)]
  pub fn ones(rows: i32, cols: i32, mat_type: i32) -> Result<JSMat> {
    let expr = cv_err!(Mat::ones(rows, cols, mat_type))?;
    let mat = cv_err!(expr.to_mat())?;
    Ok(JSMat { mat: Arc::new(mat) })
  }

  #[napi(factory)]
  pub fn eye(rows: i32, cols: i32, mat_type: i32) -> Result<JSMat> {
    let expr = cv_err!(Mat::eye(rows, cols, mat_type))?;
    let mat = cv_err!(expr.to_mat())?;
    Ok(JSMat { mat: Arc::new(mat) })
  }

  #[napi(factory)]
  pub fn from_buffer(rows: i32, cols: i32, mat_type: i32, buffer: Buffer) -> Result<JSMat> {
    if rows <= 0 || cols <= 0 {
      return Err(Error::new(
        Status::InvalidArg,
        "rows and cols must be positive",
      ));
    }
    // CV_MAT_DEPTH_MASK = 7 (lower 3 bits encode element type: 0=U8, 1=S8,
    // 2=U16, 3=S16, 4=S32, 5=F32, 6=F64, 7=F16)
    let depth = mat_type & 7;
    // CV_CN_SHIFT = 3, CV_CN_MAX = 512; mask is (512-1) = 0x1FF
    let channels = ((mat_type >> 3) & 0x1FF) as usize + 1;
    let depth_bytes: usize = match depth {
      0 | 1 => 1,
      2 | 3 => 2,
      4 | 5 => 4,
      6 => 8,
      7 => 2,
      _ => {
        return Err(Error::new(
          Status::InvalidArg,
          format!("unsupported mat_type depth: {depth}"),
        ))
      }
    };
    let required = rows as usize * cols as usize * depth_bytes * channels;
    if buffer.len() < required {
      return Err(Error::new(
        Status::InvalidArg,
        format!(
          "buffer too small: need {required} bytes, got {}",
          buffer.len()
        ),
      ));
    }
    // SAFETY: `mat` is a non-owning view over `buffer`'s bytes.
    // `buffer` (the napi Buffer) is kept alive on the stack for the duration of
    // this call. `copy_to` immediately deep-copies all pixel data into `owned`,
    // so `mat` (and the raw pointer) are never accessed after this block.
    let mat = unsafe {
      cv_err!(Mat::new_rows_cols_with_data_unsafe_def(
        rows,
        cols,
        mat_type,
        buffer.as_ptr() as *mut std::ffi::c_void
      ))?
    };
    let mut owned = Mat::default();
    cv_err!(mat.copy_to(&mut owned))?;
    Ok(JSMat {
      mat: Arc::new(owned),
    })
  }

  #[napi(getter)]
  pub fn rows(&self) -> i32 {
    self.mat.rows()
  }
  #[napi(getter)]
  pub fn cols(&self) -> i32 {
    self.mat.cols()
  }
  #[napi(getter)]
  pub fn channels(&self) -> i32 {
    self.mat.channels()
  }
  #[napi(getter)]
  pub fn mat_type(&self) -> i32 {
    self.mat.typ()
  }
  #[napi(getter)]
  pub fn depth(&self) -> i32 {
    self.mat.depth()
  }
  #[napi(getter)]
  pub fn empty(&self) -> bool {
    self.mat.empty()
  }
  #[napi(getter)]
  pub fn size(&self) -> Result<Size> {
    let s = cv_err!(self.mat.size())?;
    Ok(Size {
      width: s.width,
      height: s.height,
    })
  }
  #[napi(getter)]
  pub fn total(&self) -> i64 {
    self.mat.total() as i64
  }
  #[napi(getter)]
  pub fn elem_size(&self) -> Result<i64> {
    Ok(cv_err!(self.mat.elem_size())? as i64)
  }
  // Zero-copy: returned Buffer is backed by Mat's memory. Do not mutate while async task is in progress.
  #[napi(getter)]
  pub fn data(&self, env: Env) -> Result<Buffer> {
    let arc = Arc::clone(&self.mat);
    let (ptr, len) = {
      let bytes = cv_err!(arc.data_bytes())?;
      (bytes.as_ptr() as *mut u8, bytes.len())
    };
    if len == 0 {
      return Ok(Buffer::from(vec![]));
    }
    // Safety:
    // - `arc` is an Arc<Mat> clone that keeps the underlying C++ Mat alive.
    // - `ptr` points into the Mat's contiguous pixel buffer (valid as long as
    //   the Arc is alive).
    // - `arc` is moved into the finalize hint so it is dropped only after V8
    //   GCs the returned Buffer, ensuring the pointer remains valid.
    // - Casting *const u8 → *mut u8 is required by the C API; the Mat's pixel
    //   buffer is read-write C++ memory, so this is sound.
    let slice = unsafe { BufferSlice::from_external(&env, ptr, len, arc, |_env, _arc| {})? };
    slice.into_buffer(&env)
  }

  #[napi(js_name = "clone")]
  pub fn clone_mat(&self) -> Result<JSMat> {
    let cloned = cv_err!(self.mat.try_clone())?;
    Ok(JSMat {
      mat: Arc::new(cloned),
    })
  }
  #[napi(js_name = "copyTo")]
  pub fn copy_to(&self, dst: &mut JSMat) -> Result<()> {
    // If dst.mat has no other Arc references, write into it directly (zero extra alloc).
    // Otherwise, allocate a new Mat and replace dst.mat so existing Arc holders
    // (e.g. Buffers from .data) keep pointing at the old, unmodified data.
    if let Some(inner) = Arc::get_mut(&mut dst.mat) {
      cv_err!(self.mat.copy_to(inner))?;
    } else {
      let mut new_mat = Mat::default();
      cv_err!(self.mat.copy_to(&mut new_mat))?;
      dst.mat = Arc::new(new_mat);
    }
    Ok(())
  }
  #[napi(js_name = "convertTo")]
  pub fn convert_to(
    &self,
    rtype: i32,
    alpha: Option<f64>,
    beta: Option<f64>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<ConvertToTask> {
    AsyncTask::with_optional_signal(
      ConvertToTask {
        src: self.mat.clone(),
        rtype,
        alpha: alpha.unwrap_or(1.0),
        beta: beta.unwrap_or(0.0),
      },
      abort_signal,
    )
  }
  // Always performs a full data copy since the reshaped view's lifetime is tied to self.
  #[napi(js_name = "reshape")]
  pub fn reshape(&self, cn: i32, rows: Option<i32>) -> Result<JSMat> {
    let reshaped = cv_err!(self.mat.reshape(cn, rows.unwrap_or(0)))?;
    // C++ Mat::reshape() returns an independent Mat sharing the same pixel buffer
    // (refcount +1), so no pixel copy is needed conceptually.
    // However, the opencv crate binds it as BoxedRef<'_, Mat>, adding a Rust
    // lifetime that ties the result to &self.  Since JSMat requires an owned
    // Arc<Mat>, try_clone() is the only way to obtain one without unsafe.
    // This is a known over-conservative binding in the opencv crate, not a
    // fundamental Rust limitation.
    let mat = cv_err!(reshaped.try_clone())?;
    Ok(JSMat { mat: Arc::new(mat) })
  }

  #[napi(js_name = "matchTemplate")]
  pub fn match_template(
    &self,
    template: &JSMat,
    method: TemplateMatchMode,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<MatchTemplateTask> {
    AsyncTask::with_optional_signal(
      MatchTemplateTask {
        src: self.mat.clone(),
        tpl: template.mat.clone(),
        method: method as i32,
      },
      abort_signal,
    )
  }
  #[napi(js_name = "matchTemplateAll")]
  pub fn match_template_all(
    &self,
    template: &JSMat,
    method: TemplateMatchMode,
    score: f64,
    nms_threshold: f64,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<MatchTemplateAllTask> {
    AsyncTask::with_optional_signal(
      MatchTemplateAllTask {
        src: self.mat.clone(),
        tpl: template.mat.clone(),
        method: method as i32,
        score,
        nms_threshold,
      },
      abort_signal,
    )
  }
  #[napi(js_name = "minMaxLoc")]
  pub fn min_max_loc(&self, abort_signal: Option<AbortSignal>) -> AsyncTask<MinMaxLocTask> {
    AsyncTask::with_optional_signal(
      MinMaxLocTask {
        src: self.mat.clone(),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "threshold")]
  pub fn threshold(
    &self,
    thresh: f64,
    maxval: f64,
    typ: ThresholdType,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<ThresholdTask> {
    AsyncTask::with_optional_signal(
      ThresholdTask {
        src: self.mat.clone(),
        thresh,
        maxval,
        typ: typ as i32,
      },
      abort_signal,
    )
  }
  #[napi(js_name = "cvtColor")]
  pub fn cvt_color(
    &self,
    code: ColorCode,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<CvtColorTask> {
    AsyncTask::with_optional_signal(
      CvtColorTask {
        src: self.mat.clone(),
        code: code as i32,
      },
      abort_signal,
    )
  }
  #[napi(js_name = "resize")]
  pub fn resize(
    &self,
    width: i32,
    height: i32,
    fx: Option<f64>,
    fy: Option<f64>,
    interpolation: Option<InterpolationFlag>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<ResizeTask> {
    AsyncTask::with_optional_signal(
      ResizeTask {
        src: self.mat.clone(),
        width,
        height,
        fx: fx.unwrap_or(0.0),
        fy: fy.unwrap_or(0.0),
        interpolation: interpolation.unwrap_or(InterpolationFlag::Linear) as i32,
      },
      abort_signal,
    )
  }
  #[napi(js_name = "gaussianBlur")]
  pub fn gaussian_blur(
    &self,
    ksize_width: i32,
    ksize_height: i32,
    sigma_x: f64,
    sigma_y: Option<f64>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<GaussianBlurTask> {
    AsyncTask::with_optional_signal(
      GaussianBlurTask {
        src: self.mat.clone(),
        ksize_width,
        ksize_height,
        sigma_x,
        sigma_y: sigma_y.unwrap_or(0.0),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "canny")]
  pub fn canny(
    &self,
    threshold1: f64,
    threshold2: f64,
    aperture_size: Option<i32>,
    l2gradient: Option<bool>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<CannyTask> {
    AsyncTask::with_optional_signal(
      CannyTask {
        src: self.mat.clone(),
        threshold1,
        threshold2,
        aperture_size: aperture_size.unwrap_or(3),
        l2gradient: l2gradient.unwrap_or(false),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "dilate")]
  pub fn dilate(
    &self,
    kernel: &JSMat,
    iterations: Option<i32>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<DilateTask> {
    AsyncTask::with_optional_signal(
      DilateTask {
        src: self.mat.clone(),
        kernel: kernel.mat.clone(),
        iterations: iterations.unwrap_or(1),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "erode")]
  pub fn erode(
    &self,
    kernel: &JSMat,
    iterations: Option<i32>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<ErodeTask> {
    AsyncTask::with_optional_signal(
      ErodeTask {
        src: self.mat.clone(),
        kernel: kernel.mat.clone(),
        iterations: iterations.unwrap_or(1),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "morphologyEx")]
  pub fn morphology_ex(
    &self,
    op: MorphType,
    kernel: &JSMat,
    iterations: Option<i32>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<MorphologyExTask> {
    AsyncTask::with_optional_signal(
      MorphologyExTask {
        src: self.mat.clone(),
        op: op as i32,
        kernel: kernel.mat.clone(),
        iterations: iterations.unwrap_or(1),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "warpAffine")]
  pub fn warp_affine(
    &self,
    m: &JSMat,
    dst_width: i32,
    dst_height: i32,
    flags: Option<i32>,
    border_mode: Option<BorderType>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<WarpAffineTask> {
    AsyncTask::with_optional_signal(
      WarpAffineTask {
        src: self.mat.clone(),
        m: m.mat.clone(),
        dst_width,
        dst_height,
        flags: flags.unwrap_or(opencv::imgproc::INTER_LINEAR),
        border_mode: border_mode
          .map(|b| b as i32)
          .unwrap_or(opencv::core::BORDER_CONSTANT),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "flip")]
  pub fn flip(
    &self,
    flip_code: FlipCode,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<FlipTask> {
    AsyncTask::with_optional_signal(
      FlipTask {
        src: self.mat.clone(),
        flip_code: flip_code as i32,
      },
      abort_signal,
    )
  }
  #[napi(js_name = "crop")]
  pub fn crop(
    &self,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<CropTask> {
    AsyncTask::with_optional_signal(
      CropTask {
        src: self.mat.clone(),
        x,
        y,
        width,
        height,
      },
      abort_signal,
    )
  }
  #[napi(js_name = "inRange")]
  pub fn in_range(
    &self,
    lower: crate::types::Scalar,
    upper: crate::types::Scalar,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<InRangeTask> {
    AsyncTask::with_optional_signal(
      InRangeTask {
        src: self.mat.clone(),
        lower,
        upper,
      },
      abort_signal,
    )
  }
  #[napi(js_name = "bitwiseAnd")]
  pub fn bitwise_and(
    &self,
    other: &JSMat,
    mask: Option<&JSMat>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<BitwiseAndTask> {
    AsyncTask::with_optional_signal(
      BitwiseAndTask {
        src1: self.mat.clone(),
        src2: other.mat.clone(),
        mask: mask.map(|m| m.mat.clone()),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "bitwiseOr")]
  pub fn bitwise_or(
    &self,
    other: &JSMat,
    mask: Option<&JSMat>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<BitwiseOrTask> {
    AsyncTask::with_optional_signal(
      BitwiseOrTask {
        src1: self.mat.clone(),
        src2: other.mat.clone(),
        mask: mask.map(|m| m.mat.clone()),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "bitwiseNot")]
  pub fn bitwise_not(
    &self,
    mask: Option<&JSMat>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<BitwiseNotTask> {
    AsyncTask::with_optional_signal(
      BitwiseNotTask {
        src: self.mat.clone(),
        mask: mask.map(|m| m.mat.clone()),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "bitwiseXor")]
  pub fn bitwise_xor(
    &self,
    other: &JSMat,
    mask: Option<&JSMat>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<BitwiseXorTask> {
    AsyncTask::with_optional_signal(
      BitwiseXorTask {
        src1: self.mat.clone(),
        src2: other.mat.clone(),
        mask: mask.map(|m| m.mat.clone()),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "split")]
  pub fn split(&self, abort_signal: Option<AbortSignal>) -> AsyncTask<SplitTask> {
    AsyncTask::with_optional_signal(
      SplitTask {
        src: self.mat.clone(),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "normalize")]
  pub fn normalize(
    &self,
    alpha: Option<f64>,
    beta: Option<f64>,
    norm_type: Option<NormType>,
    dtype: Option<i32>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<NormalizeTask> {
    AsyncTask::with_optional_signal(
      NormalizeTask {
        src: self.mat.clone(),
        alpha: alpha.unwrap_or(1.0),
        beta: beta.unwrap_or(0.0),
        norm_type: norm_type.map(|t| t as i32).unwrap_or(opencv::core::NORM_L2),
        dtype: dtype.unwrap_or(-1),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "addWeighted")]
  pub fn add_weighted(
    &self,
    alpha: f64,
    src2: &JSMat,
    beta: f64,
    gamma: f64,
    dtype: Option<i32>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<AddWeightedTask> {
    AsyncTask::with_optional_signal(
      AddWeightedTask {
        src1: self.mat.clone(),
        alpha,
        src2: src2.mat.clone(),
        beta,
        gamma,
        dtype: dtype.unwrap_or(-1),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "medianBlur")]
  pub fn median_blur(
    &self,
    ksize: i32,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<MedianBlurTask> {
    AsyncTask::with_optional_signal(
      MedianBlurTask {
        src: self.mat.clone(),
        ksize,
      },
      abort_signal,
    )
  }
  #[napi(js_name = "bilateralFilter")]
  pub fn bilateral_filter(
    &self,
    d: i32,
    sigma_color: f64,
    sigma_space: f64,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<BilateralFilterTask> {
    AsyncTask::with_optional_signal(
      BilateralFilterTask {
        src: self.mat.clone(),
        d,
        sigma_color,
        sigma_space,
      },
      abort_signal,
    )
  }
  #[allow(clippy::too_many_arguments)]
  #[napi(js_name = "sobel")]
  pub fn sobel(
    &self,
    ddepth: i32,
    dx: i32,
    dy: i32,
    ksize: Option<i32>,
    scale: Option<f64>,
    delta: Option<f64>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<SobelTask> {
    AsyncTask::with_optional_signal(
      SobelTask {
        src: self.mat.clone(),
        ddepth,
        dx,
        dy,
        ksize: ksize.unwrap_or(3),
        scale: scale.unwrap_or(1.0),
        delta: delta.unwrap_or(0.0),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "laplacian")]
  pub fn laplacian(
    &self,
    ddepth: i32,
    ksize: Option<i32>,
    scale: Option<f64>,
    delta: Option<f64>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<LaplacianTask> {
    AsyncTask::with_optional_signal(
      LaplacianTask {
        src: self.mat.clone(),
        ddepth,
        ksize: ksize.unwrap_or(1),
        scale: scale.unwrap_or(1.0),
        delta: delta.unwrap_or(0.0),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "equalizeHist")]
  pub fn equalize_hist(&self, abort_signal: Option<AbortSignal>) -> AsyncTask<EqualizeHistTask> {
    AsyncTask::with_optional_signal(
      EqualizeHistTask {
        src: self.mat.clone(),
      },
      abort_signal,
    )
  }
  #[allow(clippy::too_many_arguments)]
  #[napi(js_name = "copyMakeBorder")]
  pub fn copy_make_border(
    &self,
    top: i32,
    bottom: i32,
    left: i32,
    right: i32,
    border_type: BorderType,
    value: Option<Vec<f64>>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<CopyMakeBorderTask> {
    let [v0, v1, v2, v3] = vec4(value);
    AsyncTask::with_optional_signal(
      CopyMakeBorderTask {
        src: self.mat.clone(),
        top,
        bottom,
        left,
        right,
        border_type: border_type as i32,
        value: [v0, v1, v2, v3],
      },
      abort_signal,
    )
  }
  #[napi(js_name = "filter2D")]
  pub fn filter2d(
    &self,
    ddepth: i32,
    kernel: &JSMat,
    delta: Option<f64>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<Filter2DTask> {
    AsyncTask::with_optional_signal(
      Filter2DTask {
        src: self.mat.clone(),
        ddepth,
        kernel: kernel.mat.clone(),
        delta: delta.unwrap_or(0.0),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "absDiff")]
  pub fn abs_diff(
    &self,
    other: &JSMat,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<AbsDiffTask> {
    AsyncTask::with_optional_signal(
      AbsDiffTask {
        src1: self.mat.clone(),
        src2: other.mat.clone(),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "add")]
  pub fn add(&self, other: &JSMat, abort_signal: Option<AbortSignal>) -> AsyncTask<AddTask> {
    AsyncTask::with_optional_signal(
      AddTask {
        src1: self.mat.clone(),
        src2: other.mat.clone(),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "subtract")]
  pub fn subtract(
    &self,
    other: &JSMat,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<SubtractTask> {
    AsyncTask::with_optional_signal(
      SubtractTask {
        src1: self.mat.clone(),
        src2: other.mat.clone(),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "multiply")]
  pub fn multiply(
    &self,
    other: &JSMat,
    scale: Option<f64>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<MultiplyTask> {
    AsyncTask::with_optional_signal(
      MultiplyTask {
        src1: self.mat.clone(),
        src2: other.mat.clone(),
        scale: scale.unwrap_or(1.0),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "warpPerspective")]
  pub fn warp_perspective(
    &self,
    m: &JSMat,
    dst_width: i32,
    dst_height: i32,
    flags: Option<i32>,
    border_mode: Option<BorderType>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<WarpPerspectiveTask> {
    AsyncTask::with_optional_signal(
      WarpPerspectiveTask {
        src: self.mat.clone(),
        m: m.mat.clone(),
        dst_width,
        dst_height,
        flags: flags.unwrap_or(opencv::imgproc::INTER_LINEAR),
        border_mode: border_mode
          .map(|b| b as i32)
          .unwrap_or(opencv::core::BORDER_CONSTANT),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "transpose")]
  pub fn transpose(&self, abort_signal: Option<AbortSignal>) -> AsyncTask<TransposeTask> {
    AsyncTask::with_optional_signal(
      TransposeTask {
        src: self.mat.clone(),
      },
      abort_signal,
    )
  }
  #[napi(js_name = "release")]
  pub fn release(&mut self) -> Result<()> {
    if let Some(mat) = Arc::get_mut(&mut self.mat) {
      cv_err!(unsafe { mat.release() })
    } else {
      // Mat is shared (e.g. an async task or a zero-copy .data Buffer holds a
      // reference). The underlying C++ memory is ref-counted and will be freed
      // once all Arc holders drop — this mirrors C++ OpenCV's behaviour.
      // If you need an immediate release, ensure no async tasks are running and
      // that no .data Buffer derived from this Mat is still alive.
      Err(napi::Error::from_reason(
        "Mat is shared; cannot release now — wait for all async tasks and .data Buffers to drop",
      ))
    }
  }
}

// ─── Global functions ────────────────────────────────────────────────────────

#[napi(js_name = "getRotationMatrix2D")]
pub fn get_rotation_matrix_2d(
  center_x: f64,
  center_y: f64,
  angle: f64,
  scale: f64,
) -> Result<JSMat> {
  let center = opencv::core::Point2f::new(center_x as f32, center_y as f32);
  let m = cv_err!(opencv::imgproc::get_rotation_matrix_2d(
    center, angle, scale
  ))?;
  Ok(JSMat { mat: Arc::new(m) })
}

#[napi(js_name = "hconcat")]
pub fn hconcat(mats: Vec<&JSMat>, abort_signal: Option<AbortSignal>) -> AsyncTask<ConcatTask> {
  AsyncTask::with_optional_signal(
    ConcatTask {
      mats: mats.iter().map(|m| m.mat.clone()).collect(),
      horizontal: true,
    },
    abort_signal,
  )
}

#[napi(js_name = "vconcat")]
pub fn vconcat(mats: Vec<&JSMat>, abort_signal: Option<AbortSignal>) -> AsyncTask<ConcatTask> {
  AsyncTask::with_optional_signal(
    ConcatTask {
      mats: mats.iter().map(|m| m.mat.clone()).collect(),
      horizontal: false,
    },
    abort_signal,
  )
}

#[napi(js_name = "countNonZero")]
pub fn count_non_zero(
  mat: &JSMat,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<CountNonZeroTask> {
  AsyncTask::with_optional_signal(
    CountNonZeroTask {
      src: mat.mat.clone(),
    },
    abort_signal,
  )
}

#[napi(js_name = "mean")]
pub fn mean_channels(mat: &JSMat, abort_signal: Option<AbortSignal>) -> AsyncTask<MeanTask> {
  AsyncTask::with_optional_signal(
    MeanTask {
      src: mat.mat.clone(),
    },
    abort_signal,
  )
}

#[napi(js_name = "merge")]
pub fn merge(channels: Vec<&JSMat>, abort_signal: Option<AbortSignal>) -> AsyncTask<MergeTask> {
  AsyncTask::with_optional_signal(
    MergeTask {
      channels: channels.iter().map(|m| m.mat.clone()).collect(),
    },
    abort_signal,
  )
}
