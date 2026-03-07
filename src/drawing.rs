use crate::constants::{HersheyFont, LineType};
use crate::cv_err;
use crate::mat::JSMat;
use crate::types::{nested_pts_to_cv, Point, Rect, Scalar, Size};
use napi::bindgen_prelude::*;
use opencv::imgproc;

#[napi(js_name = "drawLine")]
pub fn draw_line(
  mat: &mut JSMat,
  pt1: Point,
  pt2: Point,
  color: Scalar,
  thickness: Option<i32>,
  line_type: Option<LineType>,
) -> Result<()> {
  let inner = mat.get_mut()?;
  cv_err!(imgproc::line(
    inner,
    pt1.into(),
    pt2.into(),
    color.into(),
    thickness.unwrap_or(1),
    line_type.unwrap_or(LineType::Line8) as i32,
    0,
  ))?;
  Ok(())
}

#[napi(js_name = "drawRectangle")]
pub fn draw_rectangle(
  mat: &mut JSMat,
  rect: Rect,
  color: Scalar,
  thickness: Option<i32>,
  line_type: Option<LineType>,
) -> Result<()> {
  let inner = mat.get_mut()?;
  cv_err!(imgproc::rectangle(
    inner,
    rect.into(),
    color.into(),
    thickness.unwrap_or(1),
    line_type.unwrap_or(LineType::Line8) as i32,
    0,
  ))?;
  Ok(())
}

#[napi(js_name = "drawCircle")]
pub fn draw_circle(
  mat: &mut JSMat,
  center: Point,
  radius: i32,
  color: Scalar,
  thickness: Option<i32>,
  line_type: Option<LineType>,
) -> Result<()> {
  let inner = mat.get_mut()?;
  cv_err!(imgproc::circle(
    inner,
    center.into(),
    radius,
    color.into(),
    thickness.unwrap_or(1),
    line_type.unwrap_or(LineType::Line8) as i32,
    0,
  ))?;
  Ok(())
}

#[allow(clippy::too_many_arguments)]
#[napi(js_name = "putText")]
pub fn put_text(
  mat: &mut JSMat,
  text: String,
  org: Point,
  font_face: HersheyFont,
  font_scale: f64,
  color: Scalar,
  thickness: Option<i32>,
  line_type: Option<LineType>,
) -> Result<()> {
  let inner = mat.get_mut()?;
  cv_err!(imgproc::put_text(
    inner,
    &text,
    org.into(),
    font_face as i32,
    font_scale,
    color.into(),
    thickness.unwrap_or(1),
    line_type.unwrap_or(LineType::Line8) as i32,
    false,
  ))?;
  Ok(())
}

#[allow(clippy::too_many_arguments)]
#[napi(js_name = "drawEllipse")]
pub fn draw_ellipse(
  mat: &mut JSMat,
  center: Point,
  axes: Size,
  angle: f64,
  start_angle: f64,
  end_angle: f64,
  color: Scalar,
  thickness: Option<i32>,
  line_type: Option<LineType>,
) -> Result<()> {
  let inner = mat.get_mut()?;
  cv_err!(imgproc::ellipse(
    inner,
    center.into(),
    axes.into(),
    angle,
    start_angle,
    end_angle,
    color.into(),
    thickness.unwrap_or(1),
    line_type.unwrap_or(LineType::Line8) as i32,
    0,
  ))?;
  Ok(())
}

// ─── getTextSize ─────────────────────────────────────────────────────────────

#[napi(object)]
pub struct TextSize {
  pub width: i32,
  pub height: i32,
  pub baseline: i32,
}

#[napi(js_name = "getTextSize")]
pub fn get_text_size(
  text: String,
  font_face: HersheyFont,
  font_scale: f64,
  thickness: Option<i32>,
) -> Result<TextSize> {
  let mut baseline = 0i32;
  let size = cv_err!(imgproc::get_text_size(
    &text,
    font_face as i32,
    font_scale,
    thickness.unwrap_or(1),
    &mut baseline,
  ))?;
  Ok(TextSize {
    width: size.width,
    height: size.height,
    baseline,
  })
}

// ─── fillPoly ─────────────────────────────────────────────────────────────────

#[napi(js_name = "fillPoly")]
pub fn fill_poly(
  mat: &mut JSMat,
  pts: Vec<Vec<Point>>,
  color: Scalar,
  line_type: Option<LineType>,
) -> Result<()> {
  let pts_vec = nested_pts_to_cv(&pts);
  let inner = mat.get_mut()?;
  cv_err!(imgproc::fill_poly(
    inner,
    &pts_vec,
    color.into(),
    line_type.unwrap_or(LineType::Line8) as i32,
    0,
    opencv::core::Point::default(),
  ))?;
  Ok(())
}

// ─── drawContours ─────────────────────────────────────────────────────────────

// contour_idx: -1 draws all; max_level: None = i32::MAX (all descendants)
#[napi(js_name = "drawContours")]
pub fn draw_contours(
  mat: &mut JSMat,
  contours: Vec<Vec<Point>>,
  contour_idx: i32,
  color: Scalar,
  thickness: Option<i32>,
  line_type: Option<LineType>,
  max_level: Option<i32>,
) -> Result<()> {
  let cv_contours = nested_pts_to_cv(&contours);
  let inner = mat.get_mut()?;
  cv_err!(imgproc::draw_contours(
    inner,
    &cv_contours,
    contour_idx,
    color.into(),
    thickness.unwrap_or(1),
    line_type.unwrap_or(LineType::Line8) as i32,
    &opencv::core::no_array(),
    max_level.unwrap_or(i32::MAX),
    opencv::core::Point::default(),
  ))?;
  Ok(())
}
