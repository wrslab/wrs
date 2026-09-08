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

@group(1) @binding(0) var<uniform> u_model : mat4x4<f32>;

struct VsIn {
    @location(0) corner : vec2<f32>,
    @location(1) pos    : vec3<f32>,
    @location(2) rgb    : vec3<f32>,
};

struct VsOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) rgb : vec3<f32>,
};

@vertex
fn vs_pcd(in : VsIn) -> VsOut {
    var clip = g.proj * g.view * u_model * vec4<f32>(in.pos, 1.0);
    let offset = in.corner * g.point_size / g.viewport * clip.w;
    var out : VsOut;
    out.clip = vec4<f32>(clip.xy + offset, clip.zw);
    out.rgb = in.rgb;
    return out;
}

@fragment
fn fs_pcd(in : VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.rgb, 1.0);
}
