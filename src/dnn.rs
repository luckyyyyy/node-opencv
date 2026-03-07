use crate::constants::{DnnBackend, DnnTarget};
use crate::cv_err;
use crate::error::try_lock_named;
use crate::impl_mat_task;
use crate::impl_passthrough_task;
use crate::lock_mutex;
use crate::mat::JSMat;
use crate::types::Rect;
use crate::utils::vec4;
use napi::bindgen_prelude::*;
use opencv::{core::Vector, dnn, prelude::*};
use std::sync::{Arc, Mutex};

// ─── Shared NMS helper ───────────────────────────────────────────────────────

pub(crate) fn apply_nms(
  bboxes: &[Rect],
  scores: &[f32],
  score_threshold: f32,
  nms_threshold: f32,
) -> napi::Result<Vector<i32>> {
  let mut opencv_bboxes: Vector<opencv::core::Rect> = Vector::with_capacity(bboxes.len());
  let mut opencv_scores: Vector<f32> = Vector::with_capacity(scores.len());
  for (r, &s) in bboxes.iter().zip(scores.iter()) {
    opencv_bboxes.push(opencv::core::Rect::from(*r));
    opencv_scores.push(s);
  }
  let mut indices = Vector::new();
  cv_err!(dnn::nms_boxes(
    &opencv_bboxes,
    &opencv_scores,
    score_threshold,
    nms_threshold,
    &mut indices,
    1.0,
    0,
  ))?;
  Ok(indices)
}

// ─── NmsBoxesTask ────────────────────────────────────────────────────────────

pub(crate) struct NmsBoxesTask {
  bboxes: Vec<Rect>,
  // Stored as f64 (same precision as JS numbers) and cast to f32 on the worker
  // thread, deferring the per-element conversion cost off the event loop.
  scores: Vec<f64>,
  score_threshold: f32,
  nms_threshold: f32,
}

impl NmsBoxesTask {
  fn do_compute(&mut self) -> Result<Vec<i32>> {
    // Convert f64 scores (JS precision) to f32 on the worker thread, then
    // delegate to apply_nms to keep the NMS call in one place.
    // The Vec<f32> allocation is O(N * 4 bytes) — trivial compared to the
    // C++ Vector allocations inside apply_nms or typical NMS box counts.
    let scores_f32: Vec<f32> = self.scores.iter().map(|&s| s as f32).collect();
    let indices = apply_nms(
      &self.bboxes,
      &scores_f32,
      self.score_threshold,
      self.nms_threshold,
    )?;
    Ok(indices.iter().collect())
  }
}
impl_passthrough_task!(NmsBoxesTask, Vec<i32>);

// ─── BlobFromImageTask ───────────────────────────────────────────────────────

pub(crate) struct BlobFromImageTask {
  image: Arc<opencv::core::Mat>,
  scale_factor: f64,
  width: i32,
  height: i32,
  mean: [f64; 4],
  swap_rb: bool,
  crop: bool,
}

impl BlobFromImageTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let blob = cv_err!(dnn::blob_from_image(
      &*self.image,
      self.scale_factor,
      opencv::core::Size::new(self.width, self.height),
      opencv::core::Scalar::new(self.mean[0], self.mean[1], self.mean[2], self.mean[3]),
      self.swap_rb,
      self.crop,
      opencv::core::CV_32F,
    ))?;
    Ok(JSMat {
      mat: Arc::new(blob),
    })
  }
}
impl_mat_task!(BlobFromImageTask);

#[allow(clippy::too_many_arguments)]
#[napi(js_name = "blobFromImage")]
pub fn blob_from_image(
  image: &JSMat,
  scale_factor: Option<f64>,
  width: Option<i32>,
  height: Option<i32>,
  mean: Option<Vec<f64>>,
  swap_rb: Option<bool>,
  crop: Option<bool>,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<BlobFromImageTask> {
  AsyncTask::with_optional_signal(
    BlobFromImageTask {
      image: image.mat.clone(),
      scale_factor: scale_factor.unwrap_or(1.0),
      width: width.unwrap_or(0),
      height: height.unwrap_or(0),
      mean: vec4(mean),
      swap_rb: swap_rb.unwrap_or(false),
      crop: crop.unwrap_or(false),
    },
    abort_signal,
  )
}

// ─── Net ─────────────────────────────────────────────────────────────────────

pub(crate) struct NetRunTask {
  inner: Arc<Mutex<dnn::Net>>,
  blob: Arc<opencv::core::Mat>,
  input_name: String,
  output_name: String,
}

fn net_set_input(
  net: &mut dnn::Net,
  blob: &opencv::core::Mat,
  input_name: &str,
) -> napi::Result<()> {
  cv_err!(net.set_input(blob, input_name, 1.0, opencv::core::Scalar::default(),))
}

impl NetRunTask {
  fn do_compute(&mut self) -> Result<JSMat> {
    let mut net = lock_mutex!(self.inner, "Net")?;
    net_set_input(&mut net, &*self.blob, &self.input_name)?;
    let mat = cv_err!(net.forward_single(&self.output_name))?;
    Ok(JSMat { mat: Arc::new(mat) })
  }
}
impl_mat_task!(NetRunTask);

pub(crate) struct NetRunMultipleTask {
  inner: Arc<Mutex<dnn::Net>>,
  blob: Arc<opencv::core::Mat>,
  input_name: String,
  output_names: Vec<String>,
}

impl NetRunMultipleTask {
  fn do_compute(&mut self) -> Result<Vec<JSMat>> {
    let mut net = lock_mutex!(self.inner, "Net")?;
    net_set_input(&mut net, &*self.blob, &self.input_name)?;
    let mut blobs: opencv::core::Vector<opencv::core::Mat> =
      opencv::core::Vector::with_capacity(self.output_names.len());
    let mut names: opencv::core::Vector<String> =
      opencv::core::Vector::with_capacity(self.output_names.len());
    for n in &self.output_names {
      names.push(n.as_str());
    }
    cv_err!(net.forward(&mut blobs, &names))?;
    Ok(
      blobs
        .into_iter()
        .map(|m| JSMat { mat: Arc::new(m) })
        .collect(),
    )
  }
}
impl_passthrough_task!(NetRunMultipleTask, Vec<JSMat>);

// Tries to acquire the Net mutex for synchronous calls, returning a clear
// error if the net is currently in use by an async run() task.
fn try_lock_net<'a>(
  inner: &'a Mutex<dnn::Net>,
  caller: &'a str,
) -> napi::Result<std::sync::MutexGuard<'a, dnn::Net>> {
  try_lock_named(inner, "Net", caller)
}

#[napi]
pub struct Net {
  inner: Arc<Mutex<dnn::Net>>,
}

#[napi]
impl Net {
  #[napi(factory, js_name = "readNetFromOnnx")]
  pub fn read_net_from_onnx(path: String) -> Result<Self> {
    let net = cv_err!(dnn::read_net_from_onnx(&path))?;
    Ok(Net {
      inner: Arc::new(Mutex::new(net)),
    })
  }

  #[napi(factory, js_name = "readNetFromCaffe")]
  pub fn read_net_from_caffe(proto_path: String, model_path: String) -> Result<Self> {
    let net = cv_err!(dnn::read_net_from_caffe(&proto_path, &model_path))?;
    Ok(Net {
      inner: Arc::new(Mutex::new(net)),
    })
  }

  #[napi(factory, js_name = "readNetFromDarknet")]
  pub fn read_net_from_darknet(cfg_path: String, weights_path: String) -> Result<Self> {
    let net = cv_err!(dnn::read_net_from_darknet(&cfg_path, &weights_path))?;
    Ok(Net {
      inner: Arc::new(Mutex::new(net)),
    })
  }

  #[napi(js_name = "setPreferableBackend")]
  pub fn set_preferable_backend(&self, backend_id: DnnBackend) -> Result<()> {
    let mut net = try_lock_net(&self.inner, "setPreferableBackend")?;
    cv_err!(net.set_preferable_backend(backend_id as i32))
  }

  #[napi(js_name = "setPreferableTarget")]
  pub fn set_preferable_target(&self, target_id: DnnTarget) -> Result<()> {
    let mut net = try_lock_net(&self.inner, "setPreferableTarget")?;
    cv_err!(net.set_preferable_target(target_id as i32))
  }

  #[napi(js_name = "getUnconnectedOutLayersNames")]
  pub fn get_unconnected_out_layers_names(&self) -> Result<Vec<String>> {
    let net = try_lock_net(&self.inner, "getUnconnectedOutLayersNames")?;
    let names = cv_err!(net.get_unconnected_out_layers_names())?;
    Ok(names.iter().map(|s| s.to_string()).collect())
  }

  #[napi(js_name = "run")]
  pub fn run(
    &self,
    blob: &JSMat,
    input_name: Option<String>,
    output_name: Option<String>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<NetRunTask> {
    AsyncTask::with_optional_signal(
      NetRunTask {
        inner: self.inner.clone(),
        blob: blob.mat.clone(),
        input_name: input_name.unwrap_or_default(),
        output_name: output_name.unwrap_or_default(),
      },
      abort_signal,
    )
  }

  #[napi(js_name = "runMultiple")]
  pub fn run_multiple(
    &self,
    blob: &JSMat,
    input_name: Option<String>,
    output_names: Vec<String>,
    abort_signal: Option<AbortSignal>,
  ) -> AsyncTask<NetRunMultipleTask> {
    AsyncTask::with_optional_signal(
      NetRunMultipleTask {
        inner: self.inner.clone(),
        blob: blob.mat.clone(),
        input_name: input_name.unwrap_or_default(),
        output_names,
      },
      abort_signal,
    )
  }
}

#[napi(js_name = "nmsBoxes")]
pub fn nms_boxes(
  bboxes: Vec<Rect>,
  scores: Vec<f64>,
  score_threshold: f64,
  nms_threshold: f64,
  abort_signal: Option<AbortSignal>,
) -> AsyncTask<NmsBoxesTask> {
  AsyncTask::with_optional_signal(
    NmsBoxesTask {
      bboxes,
      scores,
      score_threshold: score_threshold as f32,
      nms_threshold: nms_threshold as f32,
    },
    abort_signal,
  )
}
