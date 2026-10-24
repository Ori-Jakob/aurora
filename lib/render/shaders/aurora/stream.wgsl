// Mirrors gfx::DispatchParams in recording.hpp byte for byte; the two change together
struct AuroraDispatch {
  output: vec4<u32>,
  inputs: array<vec4<u32>, 4>,
  groups: vec4<u32>,
  elements: vec4<u32>,
  reserved: vec4<u32>,
  user: array<vec4<u32>, 8>,
}
@group(1) @binding(0) var<storage, read_write> aurora_stream: array<u32>;
@group(1) @binding(1) var<uniform> aurora_dispatch: AuroraDispatch;
//#if AURORA_STATE_ATOMIC
@group(2) @binding(0) var<storage, read_write> aurora_state: array<atomic<u32>>;
//#else
@group(2) @binding(0) var<storage, read_write> aurora_state: array<u32>;
//#endif
override aurora_wg_x: u32 = 64u;
override aurora_wg_y: u32 = 1u;
override aurora_wg_z: u32 = 1u;

const AURORA_STREAM_MAX_INPUTS: u32 = 8u;

fn stream_input(slot: u32) -> vec2<u32> {
  if (slot >= AURORA_STREAM_MAX_INPUTS || slot >= aurora_dispatch.groups.w) {
    return vec2<u32>(0u, 0u);
  }
  let pair = aurora_dispatch.inputs[slot / 2u];
  if ((slot & 1u) == 0u) {
    return pair.xy;
  }
  return pair.zw;
}

fn stream_load(slot: u32, i: u32) -> u32 {
  let input = stream_input(slot);
  if (i >= input.y) {
    return 0u;
  }
  let word = input.x + i;
  if (word < input.x || word >= arrayLength(&aurora_stream)) {
    return 0u;
  }
  return aurora_stream[word];
}

fn stream_store(i: u32, v: u32) {
  if (i >= aurora_dispatch.output.y) {
    return;
  }
  let word = aurora_dispatch.output.x + i;
  if (word < aurora_dispatch.output.x || word >= arrayLength(&aurora_stream)) {
    return;
  }
  aurora_stream[word] = v;
}

fn stream_active(id: vec3<u32>) -> bool {
  return all(id < aurora_dispatch.elements.xyz);
}

fn stream_index(id: vec3<u32>) -> u32 {
  let width = aurora_dispatch.groups.x * aurora_wg_x;
  let height = aurora_dispatch.groups.y * aurora_wg_y;
  return (id.z * height + id.y) * width + id.x;
}

//#if AURORA_STATE_ATOMIC
fn state_load(i: u32) -> u32 {
  if (i >= arrayLength(&aurora_state)) {
    return 0u;
  }
  return atomicLoad(&aurora_state[i]);
}

fn state_store(i: u32, v: u32) {
  if (i >= arrayLength(&aurora_state)) {
    return;
  }
  atomicStore(&aurora_state[i], v);
}

fn state_add(i: u32, v: u32) -> u32 {
  if (i >= arrayLength(&aurora_state)) {
    return 0u;
  }
  return atomicAdd(&aurora_state[i], v);
}
//#else
fn state_load(i: u32) -> u32 {
  if (i >= arrayLength(&aurora_state)) {
    return 0u;
  }
  return aurora_state[i];
}

fn state_store(i: u32, v: u32) {
  if (i >= arrayLength(&aurora_state)) {
    return;
  }
  aurora_state[i] = v;
}
//#endif
