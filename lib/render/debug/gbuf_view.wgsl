//#import aurora/frame
//#import aurora/gbuffer

//#if SOURCE_UINT
@group(1) @binding(0) var view_source: texture_2d<u32>;
//#else
@group(1) @binding(0) var view_source: texture_2d<f32>;
//#endif

override VIEW_MODE: u32 = 0u;

const VIEW_GREY = vec3f(0.25, 0.25, 0.25);

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

//#if SOURCE_UINT
fn view_color(texel: vec2u) -> vec3f {
  let rec = textureLoad(view_source, texel, 0);
  let fog = gbuf_fog_decode(rec);
  let empty = all(rec == vec4u(0u));
  switch VIEW_MODE {
    case 2u: {
      return select(vec3f(fog.q), vec3f(0.0, 0.0, 0.25), empty);
    }
    case 3u: {
      return fog.fc;
    }
    default: {
      if (empty) {
        return VIEW_GREY;
      }
      return select(vec3f(0.0, 0.8, 0.0), vec3f(1.0, 0.5, 0.0), fog.lock);
    }
  }
}
//#else
fn finite2(v: vec2f) -> bool {
  let bits = bitcast<vec2u>(v) & vec2u(0x7f800000u);
  return all(bits != vec2u(0x7f800000u));
}

fn view_color(texel: vec2u) -> vec3f {
  let value = textureLoad(view_source, texel, 0);
  switch VIEW_MODE {
    case 0u: {
      if (!finite2(value.xy)) {
        return vec3f(1.0, 0.0, 1.0);
      }
      let px = gbuf_motion_pixels(value.xy, vec2f(aurora_frame.efb_size));
      return vec3f(saturate(0.5 + px / 32.0), 0.5);
    }
    case 1u: {
      return vec3f(value.r);
    }
    case 5u: {
      return value.rgb;
    }
    default: {
      var colors = array<vec3f, 4>(VIEW_GREY, vec3f(0.0, 0.3, 1.0), vec3f(0.0, 0.9, 0.9), vec3f(0.0, 0.8, 0.0));
      return colors[min(gbuf_receiver_code(value.a), 3u)];
    }
  }
}

@fragment
fn fs_blit(@builtin(position) position: vec4f) -> @location(0) vec4f {
  return vec4f(textureLoad(view_source, vec2u(position.xy), 0).rgb, 1.0);
}
//#endif

@fragment
fn fs_view(@builtin(position) position: vec4f) -> @location(0) vec4f {
  return vec4f(view_color(vec2u(position.xy)), 1.0);
}
