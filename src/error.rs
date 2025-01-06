use napi::bindgen_prelude::*;
use opencv::Error as OpenCVNativeError;

#[derive(Debug)]
pub struct OpenCVError(pub OpenCVNativeError);

impl From<OpenCVError> for Error {
    fn from(error: OpenCVError) -> Self {
        Error::new(Status::GenericFailure, error.0.to_string())
    }
}
