//#import aurora/surface

@group(1) @binding(0) var surface_rec: texture_2d<u32>;
@group(1) @binding(1) var surface_ext: texture_2d<u32>;
@group(1) @binding(2) var<storage, read> surface_table: array<vec4u>;
@group(1) @binding(3) var surface_v1: texture_storage_2d<rgba32uint, write>;

@compute @workgroup_size(8, 8)
fn expand(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(surface_v1);
  if (any(id.xy >= dims)) {
    return;
  }
  let rec = gbuf_surface_fetch(surface_rec, id.xy);
  let ext = gbuf_surface_fetch(surface_ext, id.xy);
  let entry = surface_table[gbuf_surface_entry_index(gbuf_surface_slot(rec), arrayLength(&surface_table))];
  textureStore(surface_v1, id.xy, gbuf_surface_v1_expand(rec, entry, ext));
}
