// Shaders for the scene renderer.
//
// Everything the shaders read lives in the single Globals uniform buffer
// below (bind group 0); WGSL has no loose uniforms.
//
// Note dpdy differentiates against framebuffer y, which grows downwards, so
// the facet-normal cross product is taken as cross(dY, dX) to come out
// pointing away from the surface.
//
// The vertex layouts that feed these entry points live in web/renderer.js.
struct Globals {
    view       : mat4x4<f32>,
    proj       : mat4x4<f32>,
    view_pos   : vec4<f32>,
    viewport   : vec2<f32>,
    point_size : f32,
    _pad       : f32,
};
@group(0) @binding(0) var<uniform> g : Globals;

// Lighting rig, in LINEAR intensities. The colour render targets are sRGB
// (window and offscreen alike), so the hardware gamma-encodes whatever the
// fragment shader returns -- shading must therefore be done in linear space,
// and vertex colours, which are authored as sRGB values, decoded on the way in.
// Tuned against the pre-sRGB look: matching mean brightness and highlights,
// while accepting the lifted shadows that gamma-correct shading implies (the
// old dark values came from skipping the encode, not from the light rig).
const KEY : f32 = 1.00;
const FILL : f32 = 0.15;
const AMBIENT : f32 = 0.012;
// Key falloff. The old half-Lambert SQUARED softened mid-angles so much that,
// once gamma encoding is applied, side-facing surfaces washed out grey; a
// steeper exponent restores the shading gradient.
const KEY_POW : f32 = 4.0;

fn srgb_to_linear(c : vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

struct VsIn {
    @location(0) pos    : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) m0     : vec4<f32>,
    @location(3) m1     : vec4<f32>,
    @location(4) m2     : vec4<f32>,
    @location(5) m3     : vec4<f32>,
    @location(6) rgba   : vec4<f32>,
};

struct VsOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) world_pos  : vec3<f32>,
    @location(1) rgba       : vec4<f32>,
};

@vertex
fn vs_mesh(in : VsIn) -> VsOut {
    let model = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    let wp = model * vec4<f32>(in.pos, 1.0);
    var out : VsOut;
    out.world_pos = wp.xyz;
    out.rgba = in.rgba;
    out.clip = g.proj * g.view * wp;
    return out;
}

// Inflate along the model-space normal by a view-dependent amount; drawn with
// front-face culling so only the back hull shows, as a halo.
@vertex
fn vs_outline(in : VsIn) -> VsOut {
    let model = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    let vp = g.proj * g.view * model;
    let ref_clip = vp * vec4<f32>(in.pos, 1.0);
    let factor = 0.001 * ref_clip.w;
    var out : VsOut;
    out.world_pos = vec3<f32>(0.0);
    out.rgba = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.clip = vp * vec4<f32>(in.pos + factor * in.normal, 1.0);
    return out;
}

@fragment
fn fs_matte(in : VsOut) -> @location(0) vec4<f32> {
    let dX = dpdx(in.world_pos);
    let dY = dpdy(in.world_pos);
    let N = normalize(cross(dY, dX));
    let eye = g.view_pos.xyz;
    let dist = length(eye);
    let L_key = normalize(eye + vec3<f32>(0.0, dist * 0.5, 0.0) - in.world_pos);
    let L_fill = normalize(eye + vec3<f32>(-dist * 0.5, 0.0, 0.0) - in.world_pos);
    let half_lambert = dot(N, L_key) * 0.5 + 0.5;
    let diff_key = pow(half_lambert, KEY_POW);
    // fill is a plain Lambert: its old +0.5 bias was a stand-in for ambient,
    // which double-counts now that ambient is a real (linear) term below.
    let diff_fill = max(dot(N, L_fill), 0.0);
    // Weights are LINEAR light intensities, not display values -- the render
    // target is sRGB, so the hardware gamma-encodes this on write. Ambient is
    // small on purpose: 0.03 linear still lands near 0.19 after encoding, which
    // reads as a proper shadow rather than the washed-out grey a display-space
    // 0.1 would give here.
    let lighting = (diff_key * KEY + diff_fill * FILL + AMBIENT)
                   * vec3<f32>(1.0, 1.0, 0.97);
    return vec4<f32>(srgb_to_linear(in.rgba.rgb) * lighting, in.rgba.a);
}

@fragment
fn fs_outline(in : VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
