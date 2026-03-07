use crate::cv_err;
use napi::bindgen_prelude::*;

pub(crate) fn vec4(v: Option<Vec<f64>>) -> [f64; 4] {
  let v = v.unwrap_or_default();
  [
    v.first().copied().unwrap_or(0.0),
    v.get(1).copied().unwrap_or(0.0),
    v.get(2).copied().unwrap_or(0.0),
    v.get(3).copied().unwrap_or(0.0),
  ]
}

#[napi(js_name = "getTickFrequency")]
pub fn get_tick_frequency() -> Result<f64> {
  cv_err!(opencv::core::get_tick_frequency())
}

#[napi(js_name = "getTickCount")]
pub fn get_tick_count() -> Result<f64> {
  cv_err!(opencv::core::get_tick_count()).map(|v| v as f64)
}

#[napi(js_name = "getBuildInformation")]
pub fn get_build_information() -> Result<String> {
  cv_err!(opencv::core::get_build_information())
}
