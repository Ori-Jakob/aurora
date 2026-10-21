// Mirrors render::indirect records in geometry/indirect.hpp byte for byte; the two change together
struct AuroraDrawArgs {
  vertex_count: u32,
  instance_count: u32,
  first_vertex: u32,
  first_instance: u32,
}

struct AuroraDrawIndexedArgs {
  index_count: u32,
  instance_count: u32,
  first_index: u32,
  base_vertex: i32,
  first_instance: u32,
}

struct AuroraDispatchArgs {
  x: u32,
  y: u32,
  z: u32,
}

const AURORA_MAX_WORKGROUPS_PER_DIMENSION: u32 = 65535u;
const AURORA_UNBOUNDED: u32 = 0xffffffffu;

fn aurora_draw_args(vertex_count: u32, instance_count: u32, first_vertex: u32, vertex_capacity: u32,
                    instance_capacity: u32) -> AuroraDrawArgs {
  if (first_vertex > vertex_capacity || vertex_count > vertex_capacity - first_vertex ||
      instance_count > instance_capacity) {
    return AuroraDrawArgs(0u, 0u, 0u, 0u);
  }
  return AuroraDrawArgs(vertex_count, instance_count, first_vertex, 0u);
}

fn aurora_draw_indexed_args(index_count: u32, instance_count: u32, first_index: u32, base_vertex: i32,
                            index_capacity: u32, instance_capacity: u32) -> AuroraDrawIndexedArgs {
  if (first_index > index_capacity || index_count > index_capacity - first_index ||
      instance_count > instance_capacity) {
    return AuroraDrawIndexedArgs(0u, 0u, 0u, 0i, 0u);
  }
  return AuroraDrawIndexedArgs(index_count, instance_count, first_index, base_vertex, 0u);
}

fn aurora_dispatch_args(x: u32, y: u32, z: u32) -> AuroraDispatchArgs {
  if (x > AURORA_MAX_WORKGROUPS_PER_DIMENSION || y > AURORA_MAX_WORKGROUPS_PER_DIMENSION ||
      z > AURORA_MAX_WORKGROUPS_PER_DIMENSION) {
    return AuroraDispatchArgs(0u, 0u, 0u);
  }
  return AuroraDispatchArgs(x, y, z);
}
