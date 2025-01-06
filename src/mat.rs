use napi::{
  bindgen_prelude::*,
  threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode},
};
use opencv::{
  core::{Mat, Point},
  imgproc::match_template,
  prelude::*,
};
use crate::OpenCVError;

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
pub struct MinMaxResult {
  pub min_val: f64,
  pub max_val: f64,
  pub min_x: i32,
  pub min_y: i32,
  pub max_x: i32,
  pub max_y: i32,
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
      let mut min_loc = Point::new(0, 0);
      let mut max_loc = Point::new(0, 0);

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
        min_x: min_loc.x,
        min_y: min_loc.y,
        max_x: max_loc.x,
        max_y: max_loc.y,
      });

      tsfn.call(Ok(js_result), ThreadsafeFunctionCallMode::Blocking);
    });

    Ok(())
  }

  #[napi]
  pub unsafe fn release(&mut self) {
    let _ = self.mat.release();
  }
}
