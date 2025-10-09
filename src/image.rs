use crate::{JSMat, OpenCVError};
use napi::bindgen_prelude::*;
use opencv::{
  core::Vector,
  imgcodecs::{imdecode, imencode, imread, IMREAD_COLOR},
  prelude::*,
};

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

/// Image encoding task
pub struct ImencodeTask {
  ext: String,
  mat: opencv::core::Mat,
}

#[napi]
impl Task for ImencodeTask {
  type Output = Buffer;
  type JsValue = Buffer;

  fn compute(&mut self) -> Result<Self::Output> {
    let mut buf = Vector::new();
    imencode(&self.ext, &self.mat, &mut buf, &Vector::new()).map_err(OpenCVError)?;
    Ok(Buffer::from(buf.to_vec()))
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

#[napi(js_name = "imencode")]
pub fn imencode_async(
  ext: String,
  #[napi(ts_arg_type = "JSMat")] mat: &JSMat,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<ImencodeTask> {
  AsyncTask::with_optional_signal(
    ImencodeTask {
      ext,
      mat: mat.mat.clone(),
    },
    abort_signal,
  )
}

/// Image reading task
pub struct ImreadTask {
  path: String,
  flags: i32,
}

#[napi]
impl Task for ImreadTask {
  type Output = JSMat;
  type JsValue = JSMat;

  fn compute(&mut self) -> Result<Self::Output> {
    let img = imread(&self.path, self.flags).map_err(OpenCVError)?;

    if img.empty() {
      return Err(Error::new(
        Status::InvalidArg,
        "Failed to read image".to_string(),
      ));
    }

    Ok(JSMat { mat: img })
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

#[napi(js_name = "imread")]
pub fn imread_async(
  path: String,
  flags: Option<i32>,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<ImreadTask> {
  let flags = flags.unwrap_or(IMREAD_COLOR);
  AsyncTask::with_optional_signal(ImreadTask { path, flags }, abort_signal)
}

/// Image decoding task
pub struct ImdecodeTask {
  buffer: Vec<u8>,
  flags: i32,
}

#[napi]
impl Task for ImdecodeTask {
  type Output = JSMat;
  type JsValue = JSMat;

  fn compute(&mut self) -> Result<Self::Output> {
    let vec: Vector<u8> = Vector::from_iter(self.buffer.clone());
    let img = imdecode(&vec, self.flags).map_err(OpenCVError)?;

    if img.empty() {
      return Err(Error::new(
        Status::InvalidArg,
        "Failed to decode image".to_string(),
      ));
    }

    Ok(JSMat { mat: img })
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

#[napi(js_name = "imdecode")]
pub fn imdecode_async(
  buffer: Buffer,
  flags: Option<i32>,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<ImdecodeTask> {
  let flags = flags.unwrap_or(IMREAD_COLOR);
  AsyncTask::with_optional_signal(
    ImdecodeTask {
      buffer: buffer.to_vec(),
      flags,
    },
    abort_signal,
  )
}
