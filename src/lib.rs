#![deny(clippy::all)]

mod constants;
mod core_funcs;
mod error;
mod image;
mod imgproc;
mod mat;
mod dnn;

#[macro_use]
extern crate napi_derive;

pub use constants::*;
pub use core_funcs::*;
pub use error::OpenCVError;
pub use image::*;
pub use imgproc::*;
pub use mat::JSMat;
pub use dnn::*;
