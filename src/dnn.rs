use napi::bindgen_prelude::*;
use opencv::{core::Vector, dnn};

#[napi(object)]
pub struct Rect {
  pub x: i32,
  pub y: i32,
  pub width: i32,
  pub height: i32,
}

/// NMS boxes task
pub struct NmsBoxesTask {
  bboxes: Vec<crate::mat::Rect>,
  scores: Vec<f64>,
  score_threshold: f64,
  nms_threshold: f64,
}

#[napi]
impl Task for NmsBoxesTask {
  type Output = Vec<i32>;
  type JsValue = Vec<i32>;

  fn compute(&mut self) -> Result<Self::Output> {
    let mut opencv_bboxes = Vector::new();
    for r in &self.bboxes {
      opencv_bboxes.push(opencv::core::Rect::new(r.x, r.y, r.width, r.height));
    }

    let mut opencv_scores = Vector::new();
    for s in &self.scores {
      opencv_scores.push(*s as f32);
    }

    let score_threshold_f32 = self.score_threshold as f32;
    let nms_threshold_f32 = self.nms_threshold as f32;

    let mut indices = Vector::new();
    dnn::nms_boxes(
      &opencv_bboxes,
      &opencv_scores,
      score_threshold_f32,
      nms_threshold_f32,
      &mut indices,
      1.0,
      0,
    )
    .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

    let mut result = Vec::new();
    for i in 0..indices.len() {
      result.push(indices.get(i).unwrap());
    }
    Ok(result)
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

#[napi]
pub struct DNN {}

#[napi]
impl DNN {
  #[napi(constructor)]
  pub fn new() -> Self {
    DNN {}
  }

  #[napi]
  pub fn nms_boxes(
    &self,
    bboxes: Vec<crate::mat::Rect>,
    scores: Vec<f64>,
    score_threshold: f64,
    nms_threshold: f64,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<NmsBoxesTask> {
    AsyncTask::with_optional_signal(
      NmsBoxesTask {
        bboxes,
        scores,
        score_threshold,
        nms_threshold,
      },
      abort_signal,
    )
  }
}
