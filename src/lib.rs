#![deny(clippy::all)]
// napi-rs pattern: Task structs (e.g. `ConvertToTask`) are implementation
// details consumed by napi's thread-pool machinery and never exposed to Rust
// callers — this crate is a cdylib with no Rust consumers.  The structs
// appear in the return type of `pub` `#[napi]` methods only because Rust
// requires a concrete return type for `AsyncTask<T>`.  Suppress the
// `private_interfaces` lint rather than making every Task struct `pub` (which
// would clutter the crate's Rust API surface for no benefit).
#![allow(private_interfaces)]

#[macro_use]
extern crate napi_derive;

mod constants;
mod dnn;
mod drawing;
mod error;
// SAFETY: `mod mat` MUST appear before `mod features` (and any other module
// containing `#[napi] impl JSMat`).  napi-derive resolves the JS class name
// at proc-macro expansion time; if the struct definition (with
// `#[napi(js_name = "Mat")]`) has not been processed first, the derived impl
// falls back to UpperCamelCase("JSMat") = "JsMat", silently registering the
// wrong name on the JS side. Alphabetical re-ordering of the lines below
// will break the binding without a compile error.
// rustfmt is disabled for module ordering via `reorder_modules = false` in rustfmt.toml.
mod mat;
mod features;
mod image;
mod types;
mod utils;
mod video;

pub use constants::*;
pub use dnn::*;
pub use drawing::*;
pub use features::*;
pub use image::*;
pub use mat::JSMat;
pub use types::*;
pub use utils::*;
pub use video::*;
