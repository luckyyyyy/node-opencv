use crate::OpenCVError;
use napi::bindgen_prelude::*;
use opencv::{core::Mat, core::Vector, dnn, imgproc::match_template, prelude::*};

#[napi(js_name = "Mat")]
pub struct JSMat {
  pub(crate) mat: Mat,
}

#[napi(object)]
pub struct Size {
  pub width: i32,
  pub height: i32,
}

#[napi(object)]
pub struct Point {
  pub x: i32,
  pub y: i32,
}
#[napi(object)]
#[derive(Clone)]
pub struct Rect {
  pub x: i32,
  pub y: i32,
  pub width: i32,
  pub height: i32,
}

#[napi(object)]
pub struct MinMaxResult {
  pub min_val: f64,
  pub max_val: f64,
  pub min_loc: Point,
  pub max_loc: Point,
}

/// Match template task
pub struct MatchTemplateTask {
  source_mat: Mat,
  template_mat: Mat,
  method: i32,
}

#[napi]
impl Task for MatchTemplateTask {
  type Output = JSMat;
  type JsValue = JSMat;

  fn compute(&mut self) -> Result<Self::Output> {
    let mut result = Mat::default();
    match_template(
      &self.source_mat,
      &self.template_mat,
      &mut result,
      self.method,
      &Mat::default(),
    )
    .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
    Ok(JSMat { mat: result })
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

/// Min max location task
pub struct MinMaxLocTask {
  source_mat: Mat,
}

#[napi]
impl Task for MinMaxLocTask {
  type Output = MinMaxResult;
  type JsValue = MinMaxResult;

  fn compute(&mut self) -> Result<Self::Output> {
    let mut min_val = 0f64;
    let mut max_val = 0f64;
    let mut min_loc = opencv::core::Point::new(0, 0);
    let mut max_loc = opencv::core::Point::new(0, 0);

    opencv::core::min_max_loc(
      &self.source_mat,
      Some(&mut min_val),
      Some(&mut max_val),
      Some(&mut min_loc),
      Some(&mut max_loc),
      &Mat::default(),
    )
    .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

    Ok(MinMaxResult {
      min_val,
      max_val,
      min_loc: Point {
        x: min_loc.x,
        y: min_loc.y,
      },
      max_loc: Point {
        x: max_loc.x,
        y: max_loc.y,
      },
    })
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

/// Threshold task
pub struct ThresholdTask {
  source_mat: Mat,
  thresh: f64,
  maxval: f64,
  typ: i32,
}

#[napi]
impl Task for ThresholdTask {
  type Output = JSMat;
  type JsValue = JSMat;

  fn compute(&mut self) -> Result<Self::Output> {
    let mut dst = Mat::default();
    opencv::imgproc::threshold(
      &self.source_mat,
      &mut dst,
      self.thresh,
      self.maxval,
      self.typ,
    )
    .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
    Ok(JSMat { mat: dst })
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

/// Match template all task
pub struct MatchTemplateAllTask {
  source_mat: Mat,
  template_mat: Mat,
  method: i32,
  score: f64,
  nms_threshold: f64,
  template_width: i32,
  template_height: i32,
}

#[napi]
impl Task for MatchTemplateAllTask {
  type Output = Vec<Rect>;
  type JsValue = Vec<Rect>;

  fn compute(&mut self) -> Result<Self::Output> {
    let mut result = Mat::default();
    match_template(
      &self.source_mat,
      &self.template_mat,
      &mut result,
      self.method,
      &Mat::default(),
    )
    .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

    let mut matches = Vec::new();
    let mut scores = Vec::new();
    let rows = result.rows();
    let cols = result.cols();

    let result_data =
      unsafe { std::slice::from_raw_parts(result.data() as *const f32, (rows * cols) as usize) };

    for i in 0..rows {
      for j in 0..cols {
        let value = result_data[(i * cols + j) as usize];
        if value >= self.score as f32 {
          matches.push(Rect {
            x: j,
            y: i,
            width: self.template_width,
            height: self.template_height,
          });
          scores.push(value as f64);
        }
      }
    }

    if !matches.is_empty() {
      let mut opencv_bboxes = Vector::new();
      for r in &matches {
        opencv_bboxes.push(opencv::core::Rect::new(r.x, r.y, r.width, r.height));
      }

      let mut opencv_scores = Vector::new();
      for s in &scores {
        opencv_scores.push(*s as f32);
      }

      let mut indices = Vector::new();
      dnn::nms_boxes(
        &opencv_bboxes,
        &opencv_scores,
        self.score as f32,
        self.nms_threshold as f32,
        &mut indices,
        1.0,
        0,
      )
      .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

      let filtered_matches: Vec<Rect> = indices
        .iter()
        .map(|i| matches[i as usize].clone())
        .collect();
      Ok(filtered_matches)
    } else {
      Ok(Vec::new())
    }
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

#[napi]
impl JSMat {
  #[napi(constructor)]
  pub fn new() -> Self {
    Self {
      mat: Mat::default(),
    }
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
  pub fn size(&self) -> Size {
    let size = self.mat.size().unwrap();
    Size {
      width: size.width,
      height: size.height,
    }
  }

  #[napi(getter)]
  pub fn data(&self) -> napi::Result<Buffer> {
    let vec: Vec<u8> = self.mat.data_bytes().map_err(OpenCVError)?.to_vec();
    Ok(Buffer::from(vec))
  }

  #[napi]
  pub fn match_template(
    &self,
    template: &JSMat,
    method: i32,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<MatchTemplateTask> {
    AsyncTask::with_optional_signal(
      MatchTemplateTask {
        source_mat: self.mat.clone(),
        template_mat: template.mat.clone(),
        method,
      },
      abort_signal,
    )
  }

  #[napi]
  pub fn min_max_loc(&self, abort_signal: Option<AbortSignal>) -> AsyncTask<MinMaxLocTask> {
    AsyncTask::with_optional_signal(
      MinMaxLocTask {
        source_mat: self.mat.clone(),
      },
      abort_signal,
    )
  }

  #[napi]
  pub fn threshold(
    &self,
    thresh: f64,
    maxval: f64,
    typ: i32,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<ThresholdTask> {
    AsyncTask::with_optional_signal(
      ThresholdTask {
        source_mat: self.mat.clone(),
        thresh,
        maxval,
        typ,
      },
      abort_signal,
    )
  }

  #[napi]
  pub fn match_template_all(
    &self,
    template: &JSMat,
    method: i32,
    score: f64,
    nms_threshold: f64,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<MatchTemplateAllTask> {
    AsyncTask::with_optional_signal(
      MatchTemplateAllTask {
        source_mat: self.mat.clone(),
        template_mat: template.mat.clone(),
        method,
        score,
        nms_threshold,
        template_width: template.cols(),
        template_height: template.rows(),
      },
      abort_signal,
    )
  }

  #[napi]
  pub unsafe fn release(&mut self) {
    let _ = self.mat.release();
  }
}
