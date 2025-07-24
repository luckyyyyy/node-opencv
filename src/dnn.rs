use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use opencv::{core::Vector, dnn};

#[napi(object)]
pub struct Rect {
  pub x: i32,
  pub y: i32,
  pub width: i32,
  pub height: i32,
}

#[napi]
pub struct DNN {}

impl Default for DNN {
  fn default() -> Self {
    Self::new()
  }
}

#[napi]
impl DNN {
  #[napi(constructor)]
  pub fn new() -> Self {
    DNN {}
  }

  #[napi]
  pub fn nms_boxes(
    &self,
    bboxes: Vec<Rect>,
    scores: Vec<f64>,
    score_threshold: f64,
    nms_threshold: f64,
    callback: JsFunction,
  ) -> Result<()> {
    let tsfn: ThreadsafeFunction<Result<Vec<i32>>> =
      callback.create_threadsafe_function(0, |ctx| Ok(vec![ctx.value]))?;

    let mut opencv_bboxes = Vector::new();
    for r in bboxes {
      opencv_bboxes.push(opencv::core::Rect::new(r.x, r.y, r.width, r.height));
    }

    let mut opencv_scores = Vector::new();
    for s in scores {
      opencv_scores.push(s as f32);
    }

    let score_threshold_f32 = score_threshold as f32;
    let nms_threshold_f32 = nms_threshold as f32;

    std::thread::spawn(move || {
      let mut indices = Vector::new();

      let js_result = dnn::nms_boxes(
        &opencv_bboxes,
        &opencv_scores,
        score_threshold_f32,
        nms_threshold_f32,
        &mut indices,
        1.0,
        0,
      )
      .map_err(|e| napi::Error::from_reason(e.to_string()))
      .map(|_| {
        let mut result = Vec::new();
        for i in 0..indices.len() {
          result.push(indices.get(i).unwrap());
        }
        result
      });

      tsfn.call(Ok(js_result), ThreadsafeFunctionCallMode::Blocking);
    });

    Ok(())
  }
}
