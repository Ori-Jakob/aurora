//#import aurora/composite
//#import aurora/gbuffer
//#import aurora/srgb
//#import aurora/tonemap

const EXPOSURE_GRID_W: u32 = 192u;
const EXPOSURE_GRID_H: u32 = 108u;
const EXPOSURE_GROUPS: u32 = 81u;
const EXPOSURE_GROUP_SIZE: u32 = 64u;
const EXPOSURE_SAMPLES: u32 = 4u;
const EXPOSURE_CANDIDATES: u32 = 32u;
const EXPOSURE_WORDS: u32 = 36u;
const EXPOSURE_WORD_WEIGHT: u32 = 32u;
const EXPOSURE_WORD_TARGET: u32 = 33u;
const EXPOSURE_WORD_COUNT: u32 = 34u;
const EXPOSURE_WORD_HDR: u32 = 35u;
const EXPOSURE_STEP_EV: f32 = 0.25;
const EXPOSURE_SNAP_STEP_EV: f32 = 0.5;
const EXPOSURE_MAX_EV: f32 = 8.0;
const EXPOSURE_KEY_LOG2: f32 = -2.4739311883324122;
const EXPOSURE_MATCH_TAU: f32 = 0.1;
const EXPOSURE_AUTO_UP_TAU: f32 = 1.0;
const EXPOSURE_AUTO_DOWN_TAU: f32 = 0.4;
const EXPOSURE_HOLD_COUNT: f32 = 207.36;
const EXPOSURE_FOG_CUT_Q7: u32 = 114u;
const EXPOSURE_Y_FLOOR: f32 = 0.0009765625;
const EXPOSURE_MODE_MATCH: u32 = 0u;
const EXPOSURE_MODE_MANUAL: u32 = 2u;
const EXPOSURE_FLAG_AGX: u32 = 1u;
const EXPOSURE_FLAG_TARGET_S: u32 = 2u;
const EXPOSURE_FLAG_SNAP: u32 = 4u;
const EXPOSURE_FLAG_RESET: u32 = 8u;
const EXPOSURE_STATE_VALID: u32 = 1u;
const EXPOSURE_STATE_MODE_MASK: u32 = 6u;
const EXPOSURE_STATE_AGX: u32 = 8u;
const EXPOSURE_STATE_SNAPPED: u32 = 16u;
const EXPOSURE_STATE_HELD: u32 = 32u;
const EXPOSURE_STATE_PINNED: u32 = 64u;
const EXPOSURE_STATE_TARGET_S: u32 = 128u;

struct ExposureParams {
  mode: u32,
  flags: u32,
  bias: f32,
  env_bias: f32,
  dt: f32,
  anchor: f32,
  pad0: u32,
  pad1: u32,
}

struct ExposureWords {
  e: f32,
  inv_e: f32,
  ratio: f32,
  one: f32,
}

struct ExposureWindow {
  center: f32,
  step: f32,
  snap: bool,
  prev_s: f32,
  history: bool,
}

@group(0) @binding(1) var<uniform> exposure_params: ExposureParams;
@group(1) @binding(0) var scene_hdr: texture_2d<f32>;
@group(1) @binding(1) var legacy_tex: texture_2d<f32>;
@group(1) @binding(2) var fog_record: texture_2d<u32>;
@group(1) @binding(3) var ao_final: texture_2d<f32>;
@group(1) @binding(4) var gx_normal: texture_2d<f32>;
@group(1) @binding(5) var state_out: texture_storage_2d<rgba32uint, write>;
@group(1) @binding(6) var<storage, read_write> partials: array<f32>;
@group(1) @binding(7) var<storage, read_write> exposure_out: ExposureWords;
@group(2) @binding(0) var state_prev: texture_2d<u32>;

var<workgroup> exposure_tree: array<array<f32, EXPOSURE_WORDS>, EXPOSURE_GROUP_SIZE>;
var<workgroup> exposure_sums: array<f32, EXPOSURE_WORDS>;

fn exposure_finite(v: vec3f) -> bool {
  return all((bitcast<vec3u>(v) & vec3u(0x7fffffffu)) < vec3u(0x7f800000u));
}

// Whole EV steps keep the fraction, so Manual is the same on every run and +-1 EV doubles or halves e exactly
fn exposure_exp2(s: f32) -> f32 {
  let whole = floor(s);
  let f = s - whole;
  return ldexp(select(exp2(f), 1.0, f == 0.0), i32(whole));
}

// The meter and the solve both call this, so the candidates they use always agree
fn exposure_window() -> ExposureWindow {
  let prev = textureLoad(state_prev, vec2i(0, 0), 0);
  let prev_s = bitcast<f32>(prev.x);
  let valid = (prev.y & EXPOSURE_STATE_VALID) != 0u && abs(prev_s) <= EXPOSURE_MAX_EV;
  let p = exposure_params;
  let mode_changed = ((prev.y & EXPOSURE_STATE_MODE_MASK) >> 1u) != p.mode;
  let tonemap_changed = ((prev.y & EXPOSURE_STATE_AGX) != 0u) != ((p.flags & EXPOSURE_FLAG_AGX) != 0u);
  let forced = (p.flags & (EXPOSURE_FLAG_SNAP | EXPOSURE_FLAG_RESET)) != 0u;
  // A held frame cannot take a snap it was asked for, so it passes the snap on to the next frame
  let owed = (prev.y & (EXPOSURE_STATE_SNAPPED | EXPOSURE_STATE_HELD)) == (EXPOSURE_STATE_SNAPPED | EXPOSURE_STATE_HELD);
  let snap = !valid || forced || mode_changed || tonemap_changed || owed;
  let edge = EXPOSURE_MAX_EV - 15.5 * EXPOSURE_STEP_EV;
  let s = select(prev_s, 0.0, !valid);
  return ExposureWindow(select(clamp(s, -edge, edge), 0.0, snap), select(EXPOSURE_STEP_EV, EXPOSURE_SNAP_STEP_EV, snap),
                        snap, s, valid);
}

fn exposure_candidate(w: ExposureWindow, k: u32) -> f32 {
  return w.center + (f32(k) - 15.5) * w.step;
}

fn exposure_sample(p: vec2i, w: ExposureWindow, agx: bool, acc: ptr<function, array<f32, EXPOSURE_WORDS>>) {
  let hdr = textureLoad(scene_hdr, composite_texel(p, textureDimensions(scene_hdr)), 0);
  let raw = textureLoad(fog_record, composite_texel(p, textureDimensions(fog_record)), 0);
  if (hdr.a < 0.5 || !exposure_finite(hdr.rgb) || (raw.a >> 1u) > EXPOSURE_FOG_CUT_Q7) {
    return;
  }
  let rec = gbuf_fog_decode(raw);
  let l = textureLoad(legacy_tex, composite_texel(p, textureDimensions(legacy_tex)), 0).rgb;
  let s = l - rec.q * rec.fc;
  if (any(s < vec3f(-2.0 / 255.0))) {
    return;
  }
  var t = composite_identity_terms();
  t.ao = composite_ao_term(ao_final, gx_normal, p, false);
  let sp = max(s, vec3f(0.0));
  let sx = select(sp * composite_legacy_scale(t), sp, (exposure_params.flags & EXPOSURE_FLAG_TARGET_S) != 0u);
  let weight = 1.0 - rec.q;
  let y_ld = max(tonemap_luminance(srgb_to_linear(saturate(sx / weight))), EXPOSURE_Y_FLOOR);
  (*acc)[EXPOSURE_WORD_WEIGHT] += weight;
  (*acc)[EXPOSURE_WORD_TARGET] += weight * log2(y_ld);
  (*acc)[EXPOSURE_WORD_COUNT] += 1.0;
  (*acc)[EXPOSURE_WORD_HDR] += weight * log2(max(tonemap_luminance(hdr.rgb), EXPOSURE_Y_FLOOR));
  if (agx) {
    let inset = tonemap_agx_inset_log2(hdr.rgb);
    for (var k = 0u; k < EXPOSURE_CANDIDATES; k++) {
      let y = tonemap_luminance(tonemap_agx_from_inset(inset + exposure_candidate(w, k)));
      (*acc)[k] += weight * log2(max(y, EXPOSURE_Y_FLOOR));
    }
  } else {
    for (var k = 0u; k < EXPOSURE_CANDIDATES; k++) {
      let y = tonemap_luminance(tonemap_pbr_neutral(hdr.rgb * exp2(exposure_candidate(w, k))));
      (*acc)[k] += weight * log2(max(y, EXPOSURE_Y_FLOOR));
    }
  }
}

@compute @workgroup_size(64)
fn meter(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) group: vec3u) {
  let w = exposure_window();
  let agx = (exposure_params.flags & EXPOSURE_FLAG_AGX) != 0u;
  let dims = vec2f(textureDimensions(scene_hdr));
  var acc: array<f32, EXPOSURE_WORDS>;
  for (var j = 0u; j < EXPOSURE_SAMPLES; j++) {
    let cell = j * (EXPOSURE_GROUPS * EXPOSURE_GROUP_SIZE) + group.x * EXPOSURE_GROUP_SIZE + li;
    let c = vec2f(f32(cell % EXPOSURE_GRID_W), f32(cell / EXPOSURE_GRID_W));
    let p = vec2i(floor((c + 0.5) * dims / vec2f(f32(EXPOSURE_GRID_W), f32(EXPOSURE_GRID_H))));
    exposure_sample(p, w, agx, &acc);
  }
  for (var k = 0u; k < EXPOSURE_WORDS; k++) {
    exposure_tree[li][k] = acc[k];
  }
  workgroupBarrier();
  // One fixed-order sum per word instead of atomics or a barrier per tree level, so the state is bit-identical
  if (li < EXPOSURE_WORDS) {
    var sum = 0.0;
    for (var i = 0u; i < EXPOSURE_GROUP_SIZE; i++) {
      sum += exposure_tree[i][li];
    }
    partials[group.x * EXPOSURE_WORDS + li] = sum;
  }
}

fn exposure_f(k: u32, weight: f32, t: f32) -> f32 {
  return exposure_sums[k] / weight - t;
}

@compute @workgroup_size(64)
fn solve(@builtin(local_invocation_index) li: u32) {
  let p = exposure_params;
  // Partials are unwritten under Manual, so reading them would make the state differ between runs
  if (p.mode != EXPOSURE_MODE_MANUAL && li < EXPOSURE_WORDS) {
    var sum = 0.0;
    // Nine independent loads per step keep the group-order sum from waiting on one load at a time
    for (var g = 0u; g < EXPOSURE_GROUPS; g += 9u) {
      var batch: array<f32, 9>;
      for (var j = 0u; j < 9u; j++) {
        batch[j] = partials[(g + j) * EXPOSURE_WORDS + li];
      }
      for (var j = 0u; j < 9u; j++) {
        sum += batch[j];
      }
    }
    exposure_sums[li] = sum;
  }
  workgroupBarrier();
  if (li != 0u) {
    return;
  }
  let w = exposure_window();
  var s_t = 0.0;
  var weight = 0.0;
  var t = 0.0;
  var f_lo = 0.0;
  var f_hi = 0.0;
  var count = 0.0;
  var held = false;
  var pinned = false;
  if (p.mode == EXPOSURE_MODE_MANUAL) {
    s_t = p.anchor + p.bias + p.env_bias;
  } else {
    weight = exposure_sums[EXPOSURE_WORD_WEIGHT];
    count = exposure_sums[EXPOSURE_WORD_COUNT];
    held = count < EXPOSURE_HOLD_COUNT || !(weight > 0.0);
    // Without a history to hold, a few samples still beat 0 EV
    if (held && (w.history || !(weight > 0.0))) {
      s_t = w.prev_s;
    } else {
      // MatchLegacy solves against the game's own image, so neither bias may move it
      t = select(EXPOSURE_KEY_LOG2 + p.bias + p.env_bias, exposure_sums[EXPOSURE_WORD_TARGET] / weight,
                 p.mode == EXPOSURE_MODE_MATCH);
      let first = exposure_f(0u, weight, t);
      let last = exposure_f(EXPOSURE_CANDIDATES - 1u, weight, t);
      if (first >= 0.0) {
        s_t = exposure_candidate(w, 0u);
        pinned = first > 0.0;
        f_lo = first;
        f_hi = first;
      } else if (last <= 0.0) {
        s_t = exposure_candidate(w, EXPOSURE_CANDIDATES - 1u);
        pinned = last < 0.0;
        f_lo = last;
        f_hi = last;
      } else {
        var k = 0u;
        for (var i = 0u; i < EXPOSURE_CANDIDATES - 1u; i++) {
          if (exposure_f(i + 1u, weight, t) > 0.0) {
            k = i;
            break;
          }
        }
        f_lo = exposure_f(k, weight, t);
        f_hi = exposure_f(k + 1u, weight, t);
        s_t = exposure_candidate(w, k) + w.step * -f_lo / (f_hi - f_lo);
      }
    }
  }
  var s = s_t;
  if (p.mode != EXPOSURE_MODE_MANUAL && !w.snap && !held) {
    // Auto opens up slower than it closes down, as eyes adapt to the dark slower than to light
    let tau = select(select(EXPOSURE_AUTO_DOWN_TAU, EXPOSURE_AUTO_UP_TAU, s_t > w.prev_s), EXPOSURE_MATCH_TAU,
                     p.mode == EXPOSURE_MODE_MATCH);
    s = w.prev_s + (s_t - w.prev_s) * (1.0 - exp(-p.dt / tau));
  }
  s = select(0.0, clamp(s, -EXPOSURE_MAX_EV, EXPOSURE_MAX_EV), (bitcast<u32>(s) & 0x7fffffffu) <= 0x7f800000u);
  var flags = EXPOSURE_STATE_VALID | ((p.mode << 1u) & EXPOSURE_STATE_MODE_MASK);
  flags |= select(0u, EXPOSURE_STATE_AGX, (p.flags & EXPOSURE_FLAG_AGX) != 0u);
  flags |= select(0u, EXPOSURE_STATE_SNAPPED, w.snap);
  flags |= select(0u, EXPOSURE_STATE_HELD, held);
  flags |= select(0u, EXPOSURE_STATE_PINNED, pinned);
  flags |= select(0u, EXPOSURE_STATE_TARGET_S, (p.flags & EXPOSURE_FLAG_TARGET_S) != 0u);
  textureStore(state_out, vec2i(0, 0), vec4u(bitcast<u32>(s), flags, bitcast<u32>(s_t), bitcast<u32>(weight)));
  textureStore(state_out, vec2i(1, 0), bitcast<vec4u>(vec4f(t, f_lo, f_hi, count)));
  exposure_out.e = exposure_exp2(s);
  exposure_out.inv_e = exposure_exp2(-s);
  exposure_out.ratio = select(exposure_exp2(s - w.prev_s), 1.0, w.snap);
  exposure_out.one = 1.0;
}
