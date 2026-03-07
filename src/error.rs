// Only for use inside Task::compute() — blocks the thread. Use try_lock_named for sync #[napi] methods.
#[macro_export]
macro_rules! lock_mutex {
  ($mutex:expr, $name:literal) => {
    $mutex.lock().map_err(|e| {
      napi::Error::new(
        napi::Status::GenericFailure,
        format!("{} mutex poisoned: {e}", $name),
      )
    })
  };
}

#[macro_export]
macro_rules! impl_passthrough_task {
  ($task:ident, $output:ty) => {
    #[napi]
    impl Task for $task {
      type Output = $output;
      type JsValue = $output;
      fn compute(&mut self) -> Result<Self::Output> {
        self.do_compute()
      }
      fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
      }
    }
  };
}

#[macro_export]
macro_rules! impl_mat_task {
  ($task:ident) => {
    $crate::impl_passthrough_task!($task, $crate::mat::JSMat);
  };
}

pub(crate) fn try_lock_named<'a, T>(
  mutex: &'a std::sync::Mutex<T>,
  name: &str,
  caller: &str,
) -> napi::Result<std::sync::MutexGuard<'a, T>> {
  mutex.try_lock().map_err(|e| match e {
    std::sync::TryLockError::WouldBlock => napi::Error::new(
      napi::Status::GenericFailure,
      format!("{name} is busy: {caller} cannot be called while an async task is in progress"),
    ),
    std::sync::TryLockError::Poisoned(e) => napi::Error::new(
      napi::Status::GenericFailure,
      format!("{name} mutex poisoned: {e}"),
    ),
  })
}

#[macro_export]
macro_rules! cv_err {
  ($e:expr) => {
    $e.map_err(|e| {
      #[allow(clippy::match_same_arms)]
      let status = match e.code {
        // StsBadArg / StsBadSize / StsUnmatchedSizes / StsUnmatchedFormats /
        // StsUnsupportedFormat / StsOutOfRange / StsBadFlag / StsBadPoint /
        // StsBadMask / BadImageSize / BadDepth / BadNumChannels
        -5 | -10 | -15 | -17 | -201 | -205 | -206 | -207 | -208 | -209 | -210 | -211 => {
          napi::Status::InvalidArg
        }
        _ => napi::Status::GenericFailure,
      };
      napi::Error::new(status, e.to_string())
    })
  };
}
