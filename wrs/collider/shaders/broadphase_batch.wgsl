// Broad phase + narrow phase as two kernels, for the batched collider.
//
// The obvious shape -- one thread per TRIANGLE PAIR, rejected by a
// triangle-vs-triangle AABB test -- does not scale here.  That test needs both
// triangles, so the pair, its thread and its six vertex transforms must all
// exist before anything can be rejected: 42.6 M threads for a gripper vs a
// bunny.  Carrying the tri-tri test in the same kernel is worse still: its
// register footprint costs occupancy for every one of those threads even when
// it never runs (measured on that shape: the same 0.58 ms/pose whether 10 155
// pairs survive the AABB test or none do).
//
// Here a triangle is tested against the OTHER ITEM's world AABB instead.  One
// triangle is enough for that, so the thread count is a sum rather than a
// product -- 23 666 instead of 42.6 M.
//
// That runs twice, over the same thread space:
//
//   `bounds` keeps no survivors, it only accumulates the AABB OF THE SURVIVORS
//   per (pose, pair, side).  The full item box is a poor filter for whichever
//   side holds the gripper -- a finger sits inside the target's box, so ~87% of
//   its triangles pass -- but the target triangles near that finger are few
//   (~1%) and their own box is tiny.
//
//   `broad` then filters against that tightened box, taken from the OTHER
//   side, and compacts what survives.
//
// Bounds are accumulated as order-preserving u32 keys (WGSL has no float
// atomics) and compared in key space, so nothing is ever decoded.
//
// `narrow` keeps the triangle-vs-triangle AABB early-out (cheap, and still
// rejects most of what broad lets through), and is the only kernel holding
// tri_tri_hit, so the two thin kernels are not taxed by its register use.

struct Params {
    n_pairs     : u32,
    n_items     : u32,
    tri_total   : u32,  // triangles across all pairs, both sides = broad's span
    row_stride  : u32,  // broad: work groups in x * workgroup size
    n_poses     : u32,
    row_stride2 : u32,  // narrow: work groups in x * workgroup size
    pose_base   : u32,  // first pose of this dispatch
    pose_span   : u32,  // narrow: poses covered by this dispatch
    eps         : f32,
    want_points : u32,  // narrow also records contact points when 1
    max_points  : u32,  // capacity of `points`
    _pad2       : f32,
};

@group(0) @binding(0)  var<storage, read>       verts     : array<vec4<f32>>;
@group(0) @binding(1)  var<storage, read>       faces     : array<vec4<u32>>;
@group(0) @binding(2)  var<storage, read>       desc      : array<vec4<u32>>;
@group(0) @binding(3)  var<storage, read>       pairs     : array<vec2<u32>>;
@group(0) @binding(4)  var<storage, read>       tfs       : array<mat4x4<f32>>;
// prefix over pairs of (f_cnt_a + f_cnt_b), for broad's pair lookup
@group(0) @binding(5)  var<storage, read>       tri_prefix : array<u32>;
// world AABB per (pose, item): min at 2i, max at 2i+1
@group(0) @binding(6)  var<storage, read>       item_aabb : array<vec4<f32>>;
// where each (pair, side) survivor list starts inside a pose's region
@group(0) @binding(7)  var<storage, read>       surv_off  : array<u32>;
// survivor count per (pose, pair, side)
@group(0) @binding(8)  var<storage, read_write> counts    : array<atomic<u32>>;
@group(0) @binding(9)  var<storage, read_write> survivors : array<u32>;
// prefix over (pose, pair) of n_a * n_b, for narrow's lookup
@group(0) @binding(10) var<storage, read>       p2_prefix : array<u32>;
@group(0) @binding(11) var<storage, read_write> flags     : array<atomic<u32>>;
@group(0) @binding(12) var<uniform>             params    : Params;
// AABB of the survivors per (pose, pair, side): 3 min keys then 3 max keys
@group(0) @binding(13) var<storage, read_write> surv_box  : array<atomic<u32>>;
// contact points, only written for a single-pose query (want_points)
@group(0) @binding(14) var<storage, read_write> points    : array<vec4<f32>>;
@group(0) @binding(15) var<storage, read_write> counter   : atomic<u32>;

// f32 -> u32 keeping the ordering, so atomicMin/Max can carry float bounds
fn order_key(x : f32) -> u32 {
    let b = bitcast<u32>(x);
    if ((b & 0x80000000u) != 0u) {
        return ~b;
    }
    return b | 0x80000000u;
}

fn key3(v : vec3<f32>) -> vec3<u32> {
    return vec3<u32>(order_key(v.x), order_key(v.y), order_key(v.z));
}

fn aabb_overlap(min_a : vec3<f32>, max_a : vec3<f32>,
                min_b : vec3<f32>, max_b : vec3<f32>) -> bool {
    return all(min_a <= max_b) && all(max_a >= min_b);
}

struct Chord {
    ok : bool,
    lo : f32,
    hi : f32,
};

// Interval where a triangle meets the planes' intersection line L: its two
// edges that pierce the OTHER plane (sign change in per-vertex distances d*)
// give two points on L, parameterised by the vertex projections q*.
fn chord_interval(d0 : f32, d1 : f32, d2 : f32,
                  q0 : f32, q1 : f32, q2 : f32, eps : f32) -> Chord {
    var out : Chord;
    out.lo = 1e30;
    out.hi = -1e30;
    var cnt : i32 = 0;
    if (d0 * d1 <= 0.0 && abs(d0 - d1) > eps) {
        let t = q0 + (q1 - q0) * (d0 / (d0 - d1));
        out.lo = min(out.lo, t);
        out.hi = max(out.hi, t);
        cnt = cnt + 1;
    }
    if (d1 * d2 <= 0.0 && abs(d1 - d2) > eps) {
        let t = q1 + (q2 - q1) * (d1 / (d1 - d2));
        out.lo = min(out.lo, t);
        out.hi = max(out.hi, t);
        cnt = cnt + 1;
    }
    if (d2 * d0 <= 0.0 && abs(d2 - d0) > eps) {
        let t = q2 + (q0 - q2) * (d2 / (d2 - d0));
        out.lo = min(out.lo, t);
        out.hi = max(out.hi, t);
        cnt = cnt + 1;
    }
    out.ok = cnt >= 2;
    return out;
}

struct Hit {
    ok : bool,
    p  : vec3<f32>,
};

fn tri_tri_hit(a0 : vec3<f32>, a1 : vec3<f32>, a2 : vec3<f32>,
               b0 : vec3<f32>, b1 : vec3<f32>, b2 : vec3<f32>,
               eps : f32) -> Hit {
    var out : Hit;
    out.ok = false;
    out.p = vec3<f32>(0.0);
    var n_a = cross(a1 - a0, a2 - a0);
    var n_b = cross(b1 - b0, b2 - b0);
    let n_a_len = length(n_a);
    let n_b_len = length(n_b);
    if (n_a_len < eps || n_b_len < eps) {
        return out;
    }
    n_a = n_a / n_a_len;
    n_b = n_b / n_b_len;
    let d_a = -dot(n_a, a0);
    let d_b = -dot(n_b, b0);
    let dist_b0 = dot(n_a, b0) + d_a;
    let dist_b1 = dot(n_a, b1) + d_a;
    let dist_b2 = dot(n_a, b2) + d_a;
    if ((dist_b0 > eps && dist_b1 > eps && dist_b2 > eps) ||
        (dist_b0 < -eps && dist_b1 < -eps && dist_b2 < -eps)) {
        return out;
    }
    let dist_a0 = dot(n_b, a0) + d_b;
    let dist_a1 = dot(n_b, a1) + d_b;
    let dist_a2 = dot(n_b, a2) + d_b;
    if ((dist_a0 > eps && dist_a1 > eps && dist_a2 > eps) ||
        (dist_a0 < -eps && dist_a1 < -eps && dist_a2 < -eps)) {
        return out;
    }
    let dir = cross(n_a, n_b);
    let dir_len = length(dir);
    if (dir_len < eps) {
        return out;  // coplanar -> false (match CPU)
    }
    let dir_n = dir / dir_len;
    let p0 = cross(d_b * n_a - d_a * n_b, dir) / (dir_len * dir_len + eps);
    let ca = chord_interval(dist_a0, dist_a1, dist_a2,
                            dot(a0 - p0, dir_n), dot(a1 - p0, dir_n),
                            dot(a2 - p0, dir_n), eps);
    if (!ca.ok) {
        return out;
    }
    let cb = chord_interval(dist_b0, dist_b1, dist_b2,
                            dot(b0 - p0, dir_n), dot(b1 - p0, dir_n),
                            dot(b2 - p0, dir_n), eps);
    if (!cb.ok) {
        return out;
    }
    let ov0 = max(ca.lo, cb.lo);
    let ov1 = min(ca.hi, cb.hi);
    if (ov0 >= ov1) {
        return out;
    }
    out.p = p0 + dir_n * (0.5 * (ov0 + ov1));
    out.ok = true;
    return out;
}

fn find_span(g : u32, n : u32, is_p2 : bool) -> u32 {
    var lo : u32 = 0u;
    var hi : u32 = n;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        var bound : u32;
        if (is_p2) {
            bound = p2_prefix[mid + 1u];
        } else {
            bound = tri_prefix[mid + 1u];
        }
        if (bound <= g) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// ------------------------------------------------- bounds / broad (shared)

struct Cand {
    ok       : bool,
    pair     : u32,
    side     : u32,
    face_idx : u32,
    tri_min  : vec3<f32>,
    tri_max  : vec3<f32>,
};

// Locate this thread's triangle, put it in world space, and reject it against
// the other item's full AABB -- everything bounds and broad have in common.
fn candidate(pose : u32, t : u32) -> Cand {
    var out : Cand;
    out.ok = false;
    let p = find_span(t, params.n_pairs, false);
    let pr = pairs[p];
    let da = desc[pr.x];
    let db = desc[pr.y];
    let local = t - tri_prefix[p];
    // the first f_cnt_a threads of a pair carry item A's triangles, the rest B's
    var own : u32;
    var other : u32;
    if (local < da.w) {
        own = pr.x;
        other = pr.y;
        out.face_idx = da.z + local;
        out.side = 0u;
    } else {
        own = pr.y;
        other = pr.x;
        out.face_idx = db.z + (local - da.w);
        out.side = 1u;
    }
    out.pair = p;
    let f = faces[out.face_idx].xyz;
    let tf = transpose(tfs[pose * params.n_items + own]);  // NumPy row-major
    let v0 = (tf * verts[f.x]).xyz;
    let v1 = (tf * verts[f.y]).xyz;
    let v2 = (tf * verts[f.z]).xyz;
    out.tri_min = min(min(v0, v1), v2);
    out.tri_max = max(max(v0, v1), v2);
    let other_slot = (pose * params.n_items + other) * 2u;
    out.ok = aabb_overlap(out.tri_min, out.tri_max,
                          item_aabb[other_slot].xyz,
                          item_aabb[other_slot + 1u].xyz);
    return out;
}

@compute @workgroup_size(256)
fn bounds(@builtin(global_invocation_id) gid3 : vec3<u32>) {
    let pose = params.pose_base + gid3.z;
    let t = gid3.y * params.row_stride + gid3.x;
    if (t >= params.tri_total) {
        return;
    }
    let c = candidate(pose, t);
    if (!c.ok) {
        return;
    }
    let base = ((pose * params.n_pairs + c.pair) * 2u + c.side) * 6u;
    let lo = key3(c.tri_min);
    let hi = key3(c.tri_max);
    atomicMin(&surv_box[base], lo.x);
    atomicMin(&surv_box[base + 1u], lo.y);
    atomicMin(&surv_box[base + 2u], lo.z);
    atomicMax(&surv_box[base + 3u], hi.x);
    atomicMax(&surv_box[base + 4u], hi.y);
    atomicMax(&surv_box[base + 5u], hi.z);
}

@compute @workgroup_size(256)
fn broad(@builtin(global_invocation_id) gid3 : vec3<u32>) {
    let pose = params.pose_base + gid3.z;
    let t = gid3.y * params.row_stride + gid3.x;
    if (t >= params.tri_total) {
        return;
    }
    let c = candidate(pose, t);
    if (!c.ok) {
        return;
    }
    // second filter: the box of the OTHER side's survivors, far tighter than
    // that side's whole item box when a gripper is sitting on a target
    let ob = ((pose * params.n_pairs + c.pair) * 2u + (1u - c.side)) * 6u;
    let omin = vec3<u32>(atomicLoad(&surv_box[ob]),
                         atomicLoad(&surv_box[ob + 1u]),
                         atomicLoad(&surv_box[ob + 2u]));
    let omax = vec3<u32>(atomicLoad(&surv_box[ob + 3u]),
                         atomicLoad(&surv_box[ob + 4u]),
                         atomicLoad(&surv_box[ob + 5u]));
    let lo = key3(c.tri_min);
    let hi = key3(c.tri_max);
    if (!(all(lo <= omax) && all(hi >= omin))) {
        return;
    }
    let cidx = (pose * params.n_pairs + c.pair) * 2u + c.side;
    let slot = atomicAdd(&counts[cidx], 1u);
    survivors[pose * params.tri_total + surv_off[c.pair * 2u + c.side] + slot] =
        c.face_idx;
}

// --------------------------------------------------------------- narrow

@compute @workgroup_size(256)
fn narrow(@builtin(global_invocation_id) gid3 : vec3<u32>) {
    // p2_prefix covers only this dispatch's pose range, so the span index is
    // local; counts and flags stay indexed by the absolute pose
    let g = gid3.y * params.row_stride2 + gid3.x;
    let n_spans = params.pose_span * params.n_pairs;
    if (g >= p2_prefix[n_spans]) {
        return;
    }
    let idx = find_span(g, n_spans, true);
    let pose = params.pose_base + idx / params.n_pairs;
    let p = idx % params.n_pairs;
    let n_b = atomicLoad(&counts[(pose * params.n_pairs + p) * 2u + 1u]);
    if (n_b == 0u) {
        return;
    }
    let local = g - p2_prefix[idx];
    let base = pose * params.tri_total;
    let fa_i = survivors[base + surv_off[p * 2u] + local / n_b];
    let fb_i = survivors[base + surv_off[p * 2u + 1u] + local % n_b];
    let pr = pairs[p];
    let ta = transpose(tfs[pose * params.n_items + pr.x]);
    let tb = transpose(tfs[pose * params.n_items + pr.y]);
    let fa = faces[fa_i].xyz;
    let fb = faces[fb_i].xyz;
    let a0 = (ta * verts[fa.x]).xyz;
    let a1 = (ta * verts[fa.y]).xyz;
    let a2 = (ta * verts[fa.z]).xyz;
    let b0 = (tb * verts[fb.x]).xyz;
    let b1 = (tb * verts[fb.y]).xyz;
    let b2 = (tb * verts[fb.z]).xyz;
    // still worth the per-triangle-pair AABB test: broad only guarantees each
    // triangle is near the other ITEM, not near this particular triangle
    if (!aabb_overlap(min(min(a0, a1), a2), max(max(a0, a1), a2),
                      min(min(b0, b1), b2), max(max(b0, b1), b2))) {
        return;
    }
    let hit = tri_tri_hit(a0, a1, a2, b0, b1, b2, params.eps);
    if (!hit.ok) {
        return;
    }
    atomicOr(&flags[pose], 1u);
    if (params.want_points == 1u) {
        let idx = atomicAdd(&counter, 1u);
        if (idx < params.max_points) {
            points[idx] = vec4<f32>(hit.p, f32(p));
        }
    }
}
