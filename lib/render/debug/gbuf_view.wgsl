//#import aurora/frame
//#import aurora/gbuffer
//#import aurora/ao
//#import aurora/surface

//#if SOURCE_UINT
@group(1) @binding(0) var view_source: texture_2d<u32>;
//#else
@group(1) @binding(0) var view_source: texture_2d<f32>;
//#endif

override VIEW_MODE: u32 = 0u;

const VIEW_GREY = vec3f(0.25, 0.25, 0.25);

struct ViewVarying {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> ViewVarying {
  let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return ViewVarying(vec4f(uv * 2.0 - 1.0, 0.0, 1.0), vec2f(uv.x, 1.0 - uv.y));
}

// AO can run below scene size and dummies are 1x1, so these rows map UV to texels
fn view_texel_uv(dims: vec2u, uv: vec2f) -> vec2u {
  return min(vec2u(uv * vec2f(dims)), dims - 1u);
}

//#if SOURCE_UINT
fn surface_class_color(id: u32) -> vec3f {
  if (id == GBUF_SURFACE_CLASS_NONE) {
    return VIEW_GREY;
  }
  if (id == GBUF_SURFACE_CLASS_DEFAULT) {
    return vec3f(0.0, 0.8, 0.0);
  }
  let hue = f32((id * 2654435761u) >> 8u) / 16777216.0;
  return saturate(abs(fract(hue + vec3f(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0);
}

fn surface_view_color(uv: vec2f) -> vec3f {
  let s = gbuf_surface_v1_decode(gbuf_surface_fetch(view_source, view_texel_uv(textureDimensions(view_source), uv)));
  switch VIEW_MODE {
    case 10u: {
      return select(s.albedo, VIEW_GREY, s.class_id == GBUF_SURFACE_CLASS_NONE);
    }
    case 11u: {
      return vec3f(s.roughness, s.metalness, s.ao);
    }
    case 12u: {
      return surface_class_color(s.class_id);
    }
    case 13u: {
      return s.ambient;
    }
    default: {
      return s.emissive / (1.0 + s.emissive);
    }
  }
}

fn view_color(texel: vec2u, uv: vec2f) -> vec3f {
  if (VIEW_MODE >= 10u) {
    return surface_view_color(uv);
  }
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

fn ao_view_color(uv: vec2f) -> vec3f {
  let value = textureLoad(view_source, view_texel_uv(textureDimensions(view_source), uv), 0);
  switch VIEW_MODE {
    case 7u: {
      return vec3f(value.r);
    }
    case 8u: {
      return vec3f(ao_raw_visibility(value));
    }
    case 15u: {
      let bounce = 3.14159265 * max(value.rgb, vec3f(0.0));
      return bounce / (1.0 + bounce);
    }
    case 16u: {
      return vec3f(saturate(value.a));
    }
    case 17u: {
      let radiance = max(value.rgb, vec3f(0.0)) / max(value.a, 1e-4);
      return radiance / (1.0 + radiance);
    }
    case 18u: {
      return value.rgb;
    }
    default: {
      if ((bitcast<u32>(value.r) & 0x7f800000u) == 0x7f800000u) {
        return vec3f(1.0, 0.0, 1.0);
      }
      return vec3f(saturate(log2(1.0 + value.r) / 18.0));
    }
  }
}

fn view_color(texel: vec2u, uv: vec2f) -> vec3f {
  if (VIEW_MODE >= 7u) {
    return ao_view_color(uv);
  }
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
  let scale = vec2f(textureDimensions(view_source)) / vec2f(aurora_frame.efb_size);
  return vec4f(textureLoad(view_source, vec2u(position.xy * scale), 0).rgb, 1.0);
}
//#endif

@fragment
fn fs_view(in: ViewVarying) -> @location(0) vec4f {
  return vec4f(view_color(vec2u(in.position.xy), in.uv), 1.0);
}
