# Scene & Geometry

`wrs` carries its own mesh geometry stack, so you rarely need trimesh or open3d.
Meshes are plain `(vertices, faces)` arrays, and `wrs.scene.geometry_ops`
(`wsgop`) operates on them directly.

## Scene objects

A `SceneObject` wraps a mesh plus a collision proxy. `from_file` loads STL/DAE
through the in-house loader; `collision_type` chooses the proxy (`MESH`,
`CAPSULE`, …); `.pos` / `.rotmat` place it; `.alpha` and
`toggle_render_collision` control rendering; `.clone()` duplicates it.

```python
--8<-- "examples/test_bunny.py"
```

## Convex hull and surface segmentation

`wrs.geom.fitting.convex_hull` builds a hull; `wrs.geom.surface.segment_surface`
groups coplanar faces into flat patches (within a normal tolerance) — the basis
for stable-placement and grasp reasoning. Here the bunny's hull is segmented and
each patch is revealed on `SPACE`:

```python
--8<-- "examples/test_segment_bunny.py"
```

Things to notice:

- `bunny.collisions[0].geom` is the underlying geometry (`.vs`, `.fs`).
- `wssop.mesh(vs, fs, rgb=...)` draws an arbitrary triangle mesh.
- `segment_surface(geom, normal_tol_deg=...)` returns lists of face indices, one
  per flat patch.

## Useful entry points

`wsgop` also offers `sample_surface`, `sample_count_from_area`,
`ray_triangles_batch_far` / `ray_shoot`, `clip_mesh`, `subdivide_once`,
`revolve`, and `icosahedron`.

## See also

- [`wrs.scene.geometry_ops`](../api/scene.md) — mesh operations.
- [`wrs.geom`](../api/geom.md) — loaders, fitting, surface, primitives.
- [Grasp planning](grasp_planning.md) — surface sampling feeds the planners.
