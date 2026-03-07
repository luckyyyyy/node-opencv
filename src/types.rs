// ─── Point conversion helpers ─────────────────────────────────────────────────

pub(crate) fn pts_to_cv(pts: &[Point]) -> opencv::core::Vector<opencv::core::Point> {
  let mut v = opencv::core::Vector::with_capacity(pts.len());
  for &p in pts {
    v.push(p.into());
  }
  v
}

pub(crate) fn nested_pts_to_cv(
  polys: &[Vec<Point>],
) -> opencv::core::Vector<opencv::core::Vector<opencv::core::Point>> {
  let mut outer = opencv::core::Vector::with_capacity(polys.len());
  for poly in polys {
    let mut inner = opencv::core::Vector::with_capacity(poly.len());
    for &p in poly {
      inner.push(p.into());
    }
    outer.push(inner);
  }
  outer
}

#[napi(object)]
#[derive(Clone, Copy, Debug)]
pub struct Size {
  pub width: i32,
  pub height: i32,
}

#[napi(object)]
#[derive(Clone, Copy, Debug)]
pub struct Point {
  pub x: i32,
  pub y: i32,
}

#[napi(object)]
#[derive(Clone, Copy, Debug)]
pub struct Rect {
  pub x: i32,
  pub y: i32,
  pub width: i32,
  pub height: i32,
}

#[napi(object)]
#[derive(Clone, Copy, Debug)]
pub struct Scalar {
  pub v0: f64,
  pub v1: f64,
  pub v2: f64,
  pub v3: f64,
}

impl From<Scalar> for opencv::core::Scalar {
  fn from(s: Scalar) -> Self {
    opencv::core::Scalar::new(s.v0, s.v1, s.v2, s.v3)
  }
}

#[napi(object)]
#[derive(Clone, Copy, Debug)]
pub struct MinMaxResult {
  pub min_val: f64,
  pub max_val: f64,
  pub min_loc: Point,
  pub max_loc: Point,
}

#[napi(object)]
#[derive(Clone, Copy, Debug)]
pub struct RotatedRect {
  pub center: PointF64,
  pub size: SizeF64,
  pub angle: f64,
}

#[napi(object)]
#[derive(Clone, Copy, Debug)]
pub struct PointF64 {
  pub x: f64,
  pub y: f64,
}

#[napi(object)]
#[derive(Clone, Copy, Debug)]
pub struct SizeF64 {
  pub width: f64,
  pub height: f64,
}

#[napi(object)]
#[derive(Clone, Copy, Debug)]
pub struct KeyPoint {
  pub pt: PointF64,
  pub size: f64,
  pub angle: f64,
  pub response: f64,
  pub octave: i32,
  pub class_id: i32,
}

// ─── opencv type conversions ─────────────────────────────────────────────────

impl From<Point> for opencv::core::Point {
  fn from(p: Point) -> Self {
    opencv::core::Point::new(p.x, p.y)
  }
}

impl From<opencv::core::Point> for Point {
  fn from(p: opencv::core::Point) -> Self {
    Point { x: p.x, y: p.y }
  }
}

impl From<Rect> for opencv::core::Rect {
  fn from(r: Rect) -> Self {
    opencv::core::Rect::new(r.x, r.y, r.width, r.height)
  }
}

impl From<opencv::core::Rect> for Rect {
  fn from(r: opencv::core::Rect) -> Self {
    Rect {
      x: r.x,
      y: r.y,
      width: r.width,
      height: r.height,
    }
  }
}

impl From<Size> for opencv::core::Size {
  fn from(s: Size) -> Self {
    opencv::core::Size::new(s.width, s.height)
  }
}

impl From<opencv::core::Size> for Size {
  fn from(s: opencv::core::Size) -> Self {
    Size {
      width: s.width,
      height: s.height,
    }
  }
}

// NOTE: f64 → f32 precision truncation (inherent in OpenCV's Point2f type).
impl From<PointF64> for opencv::core::Point2f {
  fn from(p: PointF64) -> Self {
    opencv::core::Point2f::new(p.x as f32, p.y as f32)
  }
}

// NOTE: same f64→f32 truncation as From<PointF64>.
impl From<SizeF64> for opencv::core::Size2f {
  fn from(s: SizeF64) -> Self {
    opencv::core::Size2f::new(s.width as f32, s.height as f32)
  }
}

impl From<opencv::core::Point2f> for PointF64 {
  fn from(p: opencv::core::Point2f) -> Self {
    PointF64 {
      x: p.x as f64,
      y: p.y as f64,
    }
  }
}

impl From<opencv::core::Size2f> for SizeF64 {
  fn from(s: opencv::core::Size2f) -> Self {
    SizeF64 {
      width: s.width as f64,
      height: s.height as f64,
    }
  }
}
