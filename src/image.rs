use napi::{
  bindgen_prelude::*,
  threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode},
};
use opencv::{
  core::Vector,
  imgcodecs::{imdecode, imread, IMREAD_COLOR},
  prelude::*,
};
use crate::{JSMat, OpenCVError};

#[napi]
pub fn get_tick_frequency() -> f64 {
  opencv::core::get_tick_frequency().expect("Failed to get tick frequency")
}

#[napi]
pub fn get_build_information() -> String {
  opencv::core::get_build_information().expect("Failed to get build information")
}

#[napi]
pub fn get_tick_count() -> i64 {
  opencv::core::get_tick_count().expect("Failed to get tick count")
}

#[napi(js_name = "imread")]
pub fn imread_sync(path: String, flags: Option<i32>) -> Result<JSMat> {
  let flags = flags.unwrap_or(IMREAD_COLOR);
  let img = imread(&path, flags).map_err(OpenCVError)?;

  if img.empty() {
    return Err(Error::new(
      Status::InvalidArg,
      "Failed to read image".to_string(),
    ));
  }

  Ok(JSMat { mat: img })
}

#[napi]
pub fn imread_callback(path: String, callback: JsFunction) -> Result<()> {
  let tsfn: ThreadsafeFunction<Result<JSMat>> =
    callback.create_threadsafe_function(0, |ctx| Ok(vec![ctx.value]))?;

  std::thread::spawn(move || {
    let result = match imread(&path, IMREAD_COLOR) {
      Ok(img) => {
        if img.empty() {
          Err(Error::new(
            Status::InvalidArg,
            "Failed to read image".to_string(),
          ))
        } else {
          Ok(JSMat { mat: img })
        }
      }
      Err(e) => Err(Error::from(OpenCVError(e))),
    };

    tsfn.call(Ok(result), ThreadsafeFunctionCallMode::Blocking);
  });

  Ok(())
}

#[napi(js_name = "imdecode")]
pub fn imdecode_sync(buffer: Buffer) -> Result<JSMat> {
  let buffer_vec = buffer.to_vec();
  let vec: Vector<u8> = Vector::from_iter(buffer_vec);
  let img = imdecode(&vec, IMREAD_COLOR).map_err(OpenCVError)?;
  if img.empty() {
    return Err(Error::new(
      Status::InvalidArg,
      "Failed to decode image".to_string(),
    ));
  }
  Ok(JSMat { mat: img })
}

#[napi]
pub fn imdecode_callback(buffer: Buffer, callback: JsFunction) -> Result<()> {
  let tsfn: ThreadsafeFunction<Result<JSMat>> =
    callback.create_threadsafe_function(0, |ctx| Ok(vec![ctx.value]))?;

  let buffer_vec = buffer.to_vec();

  std::thread::spawn(move || {
    let vec: Vector<u8> = Vector::from_iter(buffer_vec);
    let result = match imdecode(&vec, IMREAD_COLOR) {
      Ok(img) => {
        if img.empty() {
          Err(Error::new(
            Status::InvalidArg,
            "Failed to decode image".to_string(),
          ))
        } else {
          Ok(JSMat { mat: img })
        }
      }
      Err(e) => Err(Error::from(OpenCVError(e))),
    };

    tsfn.call(Ok(result), ThreadsafeFunctionCallMode::Blocking);
  });

  Ok(())
}
