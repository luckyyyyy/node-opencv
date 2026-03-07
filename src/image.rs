use crate::constants::ImreadFlag;
use crate::cv_err;
use crate::impl_mat_task;
use crate::impl_passthrough_task;
use crate::mat::JSMat;
use napi::bindgen_prelude::*;
use opencv::{
  core::{Mat, Vector},
  imgcodecs::{imdecode, imencode, imread, imwrite},
  prelude::*,
};
use std::sync::Arc;

// ─── AsyncTask: Imencode ─────────────────────────────────────────────────────

pub(crate) struct ImencodeTask {
  ext: String,
  mat: Arc<Mat>,
}

#[napi]
impl Task for ImencodeTask {
  // Keep the C++ Vector alive so resolve() can expose its memory to V8 without
  // an extra C++ → Rust copy.
  type Output = Vector<u8>;
  type JsValue = Buffer;

  fn compute(&mut self) -> Result<Self::Output> {
    let mut buf: Vector<u8> = Vector::new();
    cv_err!(imencode(&self.ext, &*self.mat, &mut buf, &Vector::new()))?;
    Ok(buf) // ← zero copy: returns the C++ Vector directly
  }

  fn resolve(&mut self, env: Env, output: Self::Output) -> Result<Self::JsValue> {
    let len = output.len();
    if len == 0 {
      return Ok(Buffer::from(vec![]));
    }
    // Safety:
    // - `output.as_slice()` points into the C++ std::vector buffer.
    // - `output` is moved into the finalize hint; the C++ vector is freed only
    //   after V8 GCs the returned Buffer, keeping the pointer valid.
    // - Casting *const u8 → *mut u8 matches the napi_create_external_buffer C
    //   signature; V8 does not mutate this memory.
    let ptr = output.as_slice().as_ptr() as *mut u8;
    let slice = unsafe { BufferSlice::from_external(&env, ptr, len, output, |_env, _vec| {})? };
    slice.into_buffer(&env)
  }
}

#[napi(js_name = "imencode")]
pub fn imencode_async(
  ext: String,
  mat: &JSMat,
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

// ─── AsyncTask: Imread ───────────────────────────────────────────────────────

pub(crate) struct ImreadTask {
  path: String,
  flags: i32,
}

impl ImreadTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let img = cv_err!(imread(&self.path, self.flags))?;
    if img.empty() {
      return Err(Error::new(
        Status::InvalidArg,
        format!("Failed to read image: {}", self.path),
      ));
    }
    Ok(JSMat { mat: Arc::new(img) })
  }
}
impl_mat_task!(ImreadTask);

#[napi(js_name = "imread")]
pub fn imread_async(
  path: String,
  flags: Option<ImreadFlag>,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<ImreadTask> {
  AsyncTask::with_optional_signal(
    ImreadTask {
      path,
      flags: flags.unwrap_or(ImreadFlag::Color) as i32,
    },
    abort_signal,
  )
}

// ─── AsyncTask: Imdecode ─────────────────────────────────────────────────────

pub(crate) struct ImdecodeTask {
  buffer: Vec<u8>,
  flags: i32,
}

impl ImdecodeTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    // Avoid cloning the buffer: create a non-owning Mat view over the raw bytes,
    // then pass it directly to imdecode which decodes into its own new allocation.
    let data = std::mem::take(&mut self.buffer);
    let img = {
      // SAFETY: `input_mat` borrows `data` via raw pointer.
      // It is dropped at end of this block, before `data` is freed.
      let input_mat = unsafe {
        let len = i32::try_from(data.len())
          .map_err(|_| Error::new(Status::InvalidArg, "imdecode buffer exceeds 2 GB"))?;
        cv_err!(Mat::new_rows_cols_with_data_unsafe_def(
          1,
          len,
          opencv::core::CV_8UC1,
          data.as_ptr() as *mut std::ffi::c_void,
        ))?
      };
      cv_err!(imdecode(&input_mat, self.flags))?
    };
    drop(data);
    if img.empty() {
      return Err(Error::new(
        Status::InvalidArg,
        "Failed to decode image buffer".to_string(),
      ));
    }
    Ok(JSMat { mat: Arc::new(img) })
  }
}
impl_mat_task!(ImdecodeTask);

#[napi(js_name = "imdecode")]
pub fn imdecode_async(
  buffer: Buffer,
  flags: Option<ImreadFlag>,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<ImdecodeTask> {
  AsyncTask::with_optional_signal(
    ImdecodeTask {
      buffer: buffer.into(), // one copy: napi Buffer → owned Vec<u8> (unavoidable at JS boundary); compute() is zero-copy
      flags: flags.unwrap_or(ImreadFlag::Color) as i32,
    },
    abort_signal,
  )
}

// ─── AsyncTask: Imwrite ──────────────────────────────────────────────────────

pub(crate) struct ImwriteTask {
  path: String,
  mat: Arc<Mat>,
}

impl ImwriteTask {
  fn do_compute(&mut self) -> Result<bool> {
    let ok = cv_err!(imwrite(&self.path, &*self.mat, &Vector::new()))?;
    if !ok {
      return Err(napi::Error::from_reason(format!(
        "imwrite failed: unable to write image to '{}'",
        self.path
      )));
    }
    Ok(ok)
  }
}
impl_passthrough_task!(ImwriteTask, bool);

#[napi(js_name = "imwrite")]
pub fn imwrite_async(
  path: String,
  mat: &JSMat,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<ImwriteTask> {
  AsyncTask::with_optional_signal(
    ImwriteTask {
      path,
      mat: mat.mat.clone(),
    },
    abort_signal,
  )
}
