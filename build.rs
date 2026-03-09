extern crate napi_build;

fn main() {
  napi_build::setup();

  // Re-run if PKG_CONFIG_PATH changes (user switches OpenCV version).
  println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");

  // Detect OpenCV minor version and set cfg flags for API differences.
  // OpenCV 4.11.0 added AlgorithmHint params to imgproc functions.
  let minor = pkg_config_opencv_minor();
  if minor >= 11 {
    println!("cargo:rustc-cfg=opencv_4_11_or_later");
  }
}

fn pkg_config_opencv_minor() -> u32 {
  let output = std::process::Command::new("pkg-config")
    .args(["--modversion", "opencv4"])
    .output()
    .or_else(|_| {
      std::process::Command::new("pkg-config")
        .args(["--modversion", "opencv"])
        .output()
    })
    .ok();
  output
    .and_then(|o| String::from_utf8(o.stdout).ok())
    .and_then(|s| {
      let parts: Vec<&str> = s.trim().split('.').collect();
      parts.get(1).and_then(|p| p.parse().ok())
    })
    .unwrap_or(0)
}
