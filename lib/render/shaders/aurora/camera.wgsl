// Assumes the MAIN viewport covers the target with depth range [0,1]; aurora records no viewport
struct StageCamera {
  depth: vec4f,
  offset: vec4f,
  units_per_metre: f32,
  flags: u32,
  pad0: u32,
  pad1: u32,
}

const CAMERA_VALID: u32 = 1u;
const CAMERA_FAR_DEPTH: f32 = 1e30;

fn camera_view_depth(cam: StageCamera, d: f32) -> f32 {
  if (d <= 0.0) {
    return CAMERA_FAR_DEPTH;
  }
  return clamp(cam.depth.x / (cam.depth.y - d), 0.0, CAMERA_FAR_DEPTH);
}

fn camera_view_position_fwd(cam: StageCamera, uv: vec2f, z: f32, full_dims: vec2f) -> vec3f {
  let ndc = vec2f(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y);
  let centre = cam.offset.xy + vec2f(-2.0, 2.0) * cam.offset.zw / full_dims;
  return vec3f((ndc + centre) * z / cam.depth.zw, z);
}

fn camera_gx_normal_fwd(n: vec3f) -> vec3f {
  return vec3f(n.x, n.y, -n.z);
}

fn camera_units_per_metre(cam: StageCamera) -> f32 {
  return cam.units_per_metre;
}

struct ViewRotation {
  rows: array<vec4f, 3>,
}

fn camera_view_dir_fwd(rot: ViewRotation, world: vec3f) -> vec3f {
  return vec3f(dot(rot.rows[0].xyz, world), dot(rot.rows[1].xyz, world), -dot(rot.rows[2].xyz, world));
}

fn camera_world_dir(rot: ViewRotation, fwd: vec3f) -> vec3f {
  return rot.rows[0].xyz * fwd.x + rot.rows[1].xyz * fwd.y - rot.rows[2].xyz * fwd.z;
}

// The exact inverse of camera_view_position_fwd; z of the result is 1 when the point lies in front
fn camera_project_fwd(cam: StageCamera, v_fwd: vec3f, full_dims: vec2f) -> vec3f {
  if (v_fwd.z <= 0.0) {
    return vec3f(0.0, 0.0, 0.0);
  }
  let centre = cam.offset.xy + vec2f(-2.0, 2.0) * cam.offset.zw / full_dims;
  let ndc = v_fwd.xy * cam.depth.zw / v_fwd.z - centre;
  return vec3f(0.5 * (ndc.x + 1.0), 0.5 * (1.0 - ndc.y), 1.0);
}
