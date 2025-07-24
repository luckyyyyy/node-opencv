use crate::OpenCVError;
use napi::{
  bindgen_prelude::*,
  threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode},
};
use opencv::{core::Mat, core::Vector, dnn, imgproc::match_template, prelude::*};

#[napi(js_name = "Mat")]
pub struct JSMat {
  pub(crate) mat: Mat,
}

impl Default for JSMat {
  fn default() -> Self {
    Self::new()
  }
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
  pub fn match_template_callback(
    &self,
    template: &JSMat,
    method: i32,
    callback: JsFunction,
  ) -> Result<()> {
    let tsfn: ThreadsafeFunction<Result<JSMat>> =
      callback.create_threadsafe_function(0, |ctx| Ok(vec![ctx.value]))?;

    let self_mat = self.mat.clone();
    let template_mat = template.mat.clone();

    std::thread::spawn(move || {
      let mut result = Mat::default();
      let js_result = match_template(
        &self_mat,
        &template_mat,
        &mut result,
        method,
        &Mat::default(),
      )
      .map_err(|e| napi::Error::from_reason(e.to_string()))
      .map(|_| JSMat { mat: result });

      tsfn.call(Ok(js_result), ThreadsafeFunctionCallMode::Blocking);
    });

    Ok(())
  }

  #[napi]
  pub fn min_max_loc_callback(&self, callback: JsFunction) -> Result<()> {
    let tsfn: ThreadsafeFunction<Result<MinMaxResult>> =
      callback.create_threadsafe_function(0, |ctx| Ok(vec![ctx.value]))?;

    let self_mat = self.mat.clone();

    std::thread::spawn(move || {
      let mut min_val = 0f64;
      let mut max_val = 0f64;
      let mut min_loc = opencv::core::Point::new(0, 0);
      let mut max_loc = opencv::core::Point::new(0, 0);

      let js_result = opencv::core::min_max_loc(
        &self_mat,
        Some(&mut min_val),
        Some(&mut max_val),
        Some(&mut min_loc),
        Some(&mut max_loc),
        &Mat::default(),
      )
      .map_err(|e| napi::Error::from_reason(e.to_string()))
      .map(|_| MinMaxResult {
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
      });

      tsfn.call(Ok(js_result), ThreadsafeFunctionCallMode::Blocking);
    });

    Ok(())
  }

  #[napi]
  pub fn threshold_callback(
    &self,
    thresh: f64,
    maxval: f64,
    typ: i32,
    callback: JsFunction,
  ) -> Result<()> {
    let tsfn: ThreadsafeFunction<Result<JSMat>> =
      callback.create_threadsafe_function(0, |ctx| Ok(vec![ctx.value]))?;

    let self_mat = self.mat.clone();

    std::thread::spawn(move || {
      let mut dst = Mat::default();
      let js_result = opencv::imgproc::threshold(&self_mat, &mut dst, thresh, maxval, typ)
        .map_err(|e| napi::Error::from_reason(e.to_string()))
        .map(|_| JSMat { mat: dst });

      tsfn.call(Ok(js_result), ThreadsafeFunctionCallMode::Blocking);
    });

    Ok(())
  }

  #[napi]
  pub fn match_template_all_callback(
    &self,
    template: &JSMat,
    method: i32,
    score: f64,
    nms_threshold: f64,
    callback: JsFunction,
  ) -> napi::Result<()> {
    let tsfn: ThreadsafeFunction<Vec<Rect>> =
      callback.create_threadsafe_function(0, |ctx| Ok(vec![ctx.value]))?;

    let self_mat = self.mat.clone();
    let template_mat = template.mat.clone();
    let template_width = template.cols();
    let template_height = template.rows();

    std::thread::spawn(move || {
      let mut result = Mat::default();

      if let Err(e) = match_template(
        &self_mat,
        &template_mat,
        &mut result,
        method,
        &Mat::default(),
      ) {
        let err = napi::Error::from_reason(e.to_string());
        tsfn.call(Err(err), ThreadsafeFunctionCallMode::Blocking);
        return;
      }

      let mut matches = Vec::new();
      let mut scores = Vec::new();
      let rows = result.rows();
      let cols = result.cols();

      let result_data =
        unsafe { std::slice::from_raw_parts(result.data() as *const f32, (rows * cols) as usize) };

      for i in 0..rows {
        for j in 0..cols {
          let value = result_data[(i * cols + j) as usize];
          if value >= score as f32 {
            matches.push(Rect {
              x: j,
              y: i,
              width: template_width,
              height: template_height,
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

        if let Err(e) = dnn::nms_boxes(
          &opencv_bboxes,
          &opencv_scores,
          score as f32,
          nms_threshold as f32,
          &mut indices,
          1.0,
          0,
        ) {
          let err = napi::Error::from_reason(e.to_string());
          tsfn.call(Err(err), ThreadsafeFunctionCallMode::Blocking);
          return;
        }

        let filtered_matches: Vec<Rect> = indices
          .iter()
          .map(|i| matches[i as usize].clone())
          .collect();
        // release the result mat

        tsfn.call(Ok(filtered_matches), ThreadsafeFunctionCallMode::Blocking);
      } else {
        tsfn.call(Ok(Vec::new()), ThreadsafeFunctionCallMode::Blocking);
      }
    });

    Ok(())
  }

  #[napi]
  pub fn flip(&self, flip_code: i32) -> Result<JSMat> {
    let mut dst = Mat::default();
    opencv::core::flip(&self.mat, &mut dst, flip_code)
      .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
    Ok(JSMat { mat: dst })
  }

  #[napi]
  /// # Safety
  /// This function manually releases the OpenCV Mat. Use with caution to avoid double-free errors.
  pub unsafe fn release(&mut self) {
    let _ = self.mat.release();
  }
}
