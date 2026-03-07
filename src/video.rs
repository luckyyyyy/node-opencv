use crate::cv_err;
use crate::error::try_lock_named;
use crate::impl_passthrough_task;
use crate::lock_mutex;
use crate::mat::JSMat;
use napi::bindgen_prelude::*;
use opencv::{prelude::*, videoio};
use std::sync::{Arc, Mutex};

// ─── VideoCapture ─────────────────────────────────────────────────────────────

pub(crate) struct VideoReadTask {
  inner: Arc<Mutex<videoio::VideoCapture>>,
}

impl VideoReadTask {
  fn do_compute(&mut self) -> Result<Option<JSMat>> {
    let mut cap = lock_mutex!(self.inner, "VideoCapture")?;
    let mut frame = opencv::core::Mat::default();
    let grabbed = cv_err!(cap.read(&mut frame))?;
    if grabbed && !frame.empty() {
      Ok(Some(JSMat {
        mat: Arc::new(frame),
      }))
    } else {
      Ok(None)
    }
  }
}
impl_passthrough_task!(VideoReadTask, Option<JSMat>);

#[napi]
pub struct VideoCapture {
  inner: Arc<Mutex<videoio::VideoCapture>>,
}

#[napi]
impl VideoCapture {
  #[napi(factory, js_name = "open")]
  pub fn open(source: Either<String, i32>) -> Result<Self> {
    let cap = match source {
      Either::A(path) => cv_err!(videoio::VideoCapture::from_file(&path, videoio::CAP_ANY))?,
      Either::B(index) => cv_err!(videoio::VideoCapture::new(index, videoio::CAP_ANY))?,
    };
    Ok(VideoCapture {
      inner: Arc::new(Mutex::new(cap)),
    })
  }

  #[napi(js_name = "isOpened")]
  pub fn is_opened(&self) -> Result<bool> {
    let cap = try_lock_named(&self.inner, "VideoCapture", "isOpened")?;
    cv_err!(cap.is_opened())
  }

  #[napi(js_name = "read")]
  pub fn read(&self, abort_signal: Option<AbortSignal>) -> AsyncTask<VideoReadTask> {
    AsyncTask::with_optional_signal(
      VideoReadTask {
        inner: self.inner.clone(),
      },
      abort_signal,
    )
  }

  #[napi(js_name = "get")]
  pub fn get_prop(&self, prop_id: i32) -> Result<f64> {
    let cap = try_lock_named(&self.inner, "VideoCapture", "get")?;
    cv_err!(cap.get(prop_id))
  }

  #[napi(js_name = "set")]
  pub fn set_prop(&self, prop_id: i32, value: f64) -> Result<bool> {
    let mut cap = try_lock_named(&self.inner, "VideoCapture", "set")?;
    cv_err!(cap.set(prop_id, value))
  }

  #[napi(js_name = "release")]
  pub fn release(&self) -> Result<()> {
    let mut cap = try_lock_named(&self.inner, "VideoCapture", "release")?;
    cv_err!(cap.release())
  }
}

// ─── VideoWriteTask ───────────────────────────────────────────────────────────

pub(crate) struct VideoWriteTask {
  inner: Arc<Mutex<videoio::VideoWriter>>,
  frame: Arc<opencv::core::Mat>,
}

impl VideoWriteTask {
  fn do_compute(&mut self) -> Result<()> {
    let mut w = lock_mutex!(self.inner, "VideoWriter")?;
    cv_err!(w.write(&*self.frame))
  }
}
impl_passthrough_task!(VideoWriteTask, ());

// ─── VideoWriter ─────────────────────────────────────────────────────────────

#[napi]
pub struct VideoWriter {
  inner: Arc<Mutex<videoio::VideoWriter>>,
}

#[napi]
impl VideoWriter {
  #[napi(factory, js_name = "open")]
  pub fn open(
    filename: String,
    fourcc: String,
    fps: f64,
    width: i32,
    height: i32,
    is_color: Option<bool>,
  ) -> Result<Self> {
    let chars: Vec<char> = fourcc.chars().collect();
    if chars.len() != 4 {
      return Err(napi::Error::new(
        napi::Status::InvalidArg,
        "fourcc must be exactly 4 characters".to_string(),
      ));
    }
    let code = videoio::VideoWriter::fourcc(chars[0], chars[1], chars[2], chars[3])
      .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))?;
    let writer = cv_err!(videoio::VideoWriter::new(
      &filename,
      code,
      fps,
      opencv::core::Size::new(width, height),
      is_color.unwrap_or(true),
    ))?;
    Ok(VideoWriter {
      inner: Arc::new(Mutex::new(writer)),
    })
  }

  #[napi(js_name = "isOpened")]
  pub fn is_opened(&self) -> Result<bool> {
    let w = try_lock_named(&self.inner, "VideoWriter", "isOpened")?;
    cv_err!(w.is_opened())
  }

  #[napi(js_name = "write")]
  pub fn write(
    &self,
    frame: &JSMat,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<VideoWriteTask> {
    AsyncTask::with_optional_signal(
      VideoWriteTask {
        inner: self.inner.clone(),
        frame: frame.mat.clone(),
      },
      abort_signal,
    )
  }

  #[napi(js_name = "release")]
  pub fn release(&self) -> Result<()> {
    let mut w = try_lock_named(&self.inner, "VideoWriter", "release")?;
    cv_err!(w.release())
  }
}
