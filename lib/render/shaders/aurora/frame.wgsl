// Mirrors detail::FrameUniforms in graph_layout.hpp byte for byte; the two change together
struct AuroraFrame {
  efb_size: vec2<u32>,
  output_size: vec2<u32>,
  inv_efb_size: vec2<f32>,
  inv_output_size: vec2<f32>,
  frame_index: u32,
  flags: u32,
  msaa_samples: u32,
  pad0: u32,
  reserved: array<vec4<u32>, 13>,
}
@group(0) @binding(0) var<uniform> aurora_frame: AuroraFrame;
