/**
 * WebGPU renderer for the page, mirroring wrs/viewer/render.py.
 *
 * The point of this file is what it does NOT contain: no shader source.  The
 * WGSL is fetched from the server, which serves viewer/shader.py's mesh_wgsl
 * and pcd_wgsl verbatim, so the page and the native window run the same
 * shading code instead of a hand-kept port.
 *
 * The pipeline set is render.py's:
 *
 *     outline      cull front, depth write        -- inflated black back hull
 *     solid        cull back,  depth write
 *     transparent  cull back,  no depth write, alpha blend, back-to-front
 *     pcd          no cull,    depth write        -- instanced screen-space quads
 *
 * As in render.py, uploads and draw recording are kept apart: writeBuffer runs
 * while no pass is open, then the pass only records draws.
 */
import * as math from './math.js';

const DEPTH_FORMAT = 'depth24plus';
const SAMPLE_COUNT = 4;      // SAMPLE_COUNT in viewer/world.py
const POINT_SIZE = 5.0;      // POINT_SIZE in viewer/render.py
const GLOBALS_FLOATS = 40;   // view 16 + proj 16 + view_pos 4 + viewport 2 + size 1 + pad

// _MESH_VERTEX_LAYOUT in viewer/render.py
const MESH_VERTEX_LAYOUT = [
  {
    arrayStride: 24, stepMode: 'vertex',
    attributes: [
      { format: 'float32x3', offset: 0, shaderLocation: 0 },
      { format: 'float32x3', offset: 12, shaderLocation: 1 },
    ],
  },
  {
    arrayStride: 64, stepMode: 'instance',
    attributes: [0, 16, 32, 48].map((offset, i) => (
      { format: 'float32x4', offset, shaderLocation: 2 + i })),
  },
  {
    arrayStride: 16, stepMode: 'instance',
    attributes: [{ format: 'float32x4', offset: 0, shaderLocation: 6 }],
  },
];

// _PCD_VERTEX_LAYOUT in viewer/render.py: one shared unit quad, one instance
// per point.  WGSL has no gl_PointSize, so vs_pcd expands the quad in clip
// space to keep a constant on-screen size regardless of depth.
const PCD_VERTEX_LAYOUT = [
  {
    arrayStride: 8, stepMode: 'vertex',
    attributes: [{ format: 'float32x2', offset: 0, shaderLocation: 0 }],
  },
  {
    arrayStride: 24, stepMode: 'instance',
    attributes: [
      { format: 'float32x3', offset: 0, shaderLocation: 1 },
      { format: 'float32x3', offset: 12, shaderLocation: 2 },
    ],
  },
];

const ALPHA_BLEND = {
  color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  alpha: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
};

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function createBufferWithData(device, data, usage) {
  const buffer = device.createBuffer(
    { size: data.byteLength, usage, mappedAtCreation: true });
  new data.constructor(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

/** Weave two (n, 3) float arrays into the stride-24 vertex both layouts use. */
function interleave3x2(a, b) {
  const n = a.length / 3;
  const out = new Float32Array(n * 6);
  for (let i = 0; i < n; i += 1) {
    out[i * 6 + 0] = a[i * 3 + 0];
    out[i * 6 + 1] = a[i * 3 + 1];
    out[i * 6 + 2] = a[i * 3 + 2];
    out[i * 6 + 3] = b[i * 3 + 0];
    out[i * 6 + 4] = b[i * 3 + 1];
    out[i * 6 + 5] = b[i * 3 + 2];
  }
  return out;
}

export class Renderer {
  static async create(canvas, shaders, clearColorHex) {
    if (!navigator.gpu) {
      throw new Error('WebGPU unavailable -- needs Chrome/Edge 113+, ' +
                      'Safari 26+, or Firefox 141+ on Windows');
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('no WebGPU adapter (GPU blocklisted?)');
    const device = await adapter.requestDevice();
    return new Renderer(canvas, device, shaders, clearColorHex);
  }

  constructor(canvas, device, shaders, clearColorHex) {
    this.canvas = canvas;
    this.device = device;
    this.models = new Map();
    this._targetSize = [0, 0];
    this._msaaTex = null;
    this._depthTex = null;
    this._globals = new Float32Array(GLOBALS_FLOATS);
    this._view = this._globals.subarray(0, 16);
    this._proj = this._globals.subarray(16, 32);

    // The native target is sRGB (offscreen.py pins rgba8unorm_srgb, and
    // get_preferred_format hands World the srgb surface format), so the
    // hardware gamma-encodes whatever fs_matte returns.  Configuring a
    // viewFormat here buys the page the same free encode -- which is what
    // lets the shader be shared: the WGSL writes linear and never encodes.
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.colorFormat = format + '-srgb';
    this.context = canvas.getContext('webgpu');
    this.context.configure({
      device, format, alphaMode: 'opaque', viewFormats: [this.colorFormat],
    });

    // clearValue is written as a shader output would be, i.e. linear in, sRGB
    // stored -- so the page background has to be de-encoded the same way
    // fs_matte de-encodes vertex colours.
    this.clearValue = {
      r: math.srgbToLinear(((clearColorHex >> 16) & 0xff) / 255),
      g: math.srgbToLinear(((clearColorHex >> 8) & 0xff) / 255),
      b: math.srgbToLinear((clearColorHex & 0xff) / 255),
      a: 1.0,
    };

    this._buildPipelines(shaders);
    this.quadBuf = createBufferWithData(device, new Float32Array(
      [-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]), GPUBufferUsage.VERTEX);
  }

  // ------------------------------------------------------------------ setup

  _buildPipelines(shaders) {
    const device = this.device;
    const meshModule = device.createShaderModule(
      { code: shaders.mesh, label: 'mesh' });
    const pcdModule = device.createShaderModule(
      { code: shaders.pcd, label: 'pcd' });

    this.globalsLayout = device.createBindGroupLayout({
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      }],
    });
    // pcd takes its model matrix through a uniform rather than through
    // instance attributes, so it needs a second bind group -- as in render.py.
    this.modelLayout = device.createBindGroupLayout({
      entries: [{
        binding: 0, visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform' },
      }],
    });
    this.globalsBuf = device.createBuffer({
      size: GLOBALS_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.globalsBindGroup = device.createBindGroup({
      layout: this.globalsLayout,
      entries: [{ binding: 0, resource: { buffer: this.globalsBuf } }],
    });

    const pipeline = (module, vs, fs, buffers, cull, depthWrite, blend,
                      bindGroupLayouts) => device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts }),
      vertex: { module, entryPoint: vs, buffers },
      primitive: { topology: 'triangle-list', frontFace: 'ccw', cullMode: cull },
      depthStencil: {
        format: DEPTH_FORMAT, depthWriteEnabled: depthWrite, depthCompare: 'less',
      },
      multisample: { count: SAMPLE_COUNT },
      fragment: {
        module, entryPoint: fs,
        targets: [{ format: this.colorFormat, blend }],
      },
    });

    const g = [this.globalsLayout];
    this.outlinePipeline = pipeline(meshModule, 'vs_outline', 'fs_outline',
      MESH_VERTEX_LAYOUT, 'front', true, undefined, g);
    this.solidPipeline = pipeline(meshModule, 'vs_mesh', 'fs_matte',
      MESH_VERTEX_LAYOUT, 'back', true, undefined, g);
    this.transparentPipeline = pipeline(meshModule, 'vs_mesh', 'fs_matte',
      MESH_VERTEX_LAYOUT, 'back', false, ALPHA_BLEND, g);
    this.pcdPipeline = pipeline(pcdModule, 'vs_pcd', 'fs_pcd',
      PCD_VERTEX_LAYOUT, 'none', true, undefined, [...g, this.modelLayout]);
  }

  /**
   * MSAA colour + depth, both sRGB-formatted so the resolve into the canvas
   * keeps the hardware encode.
   *
   * Known quirk: under Chrome SwiftShader fallback (a blocklisted GPU, a VM,
   * a headless run) resolving into a canvas view whose format came from
   * viewFormats silently drops that encode and the page renders roughly 1.4x
   * dark -- a Dawn bug, not a spec one.  Rendering into a texture we own
   * resolves correctly, and so does a real adapter (verified on D3D12:
   * pixel-identical to viewer.offscreen).  Not worked around here: a page
   * that has fallen back to software has bigger problems than gamma.
   */
  _ensureTargets(width, height) {
    if (this._targetSize[0] === width && this._targetSize[1] === height) return;
    this._msaaTex?.destroy();
    this._depthTex?.destroy();
    const size = { width, height };
    const usage = GPUTextureUsage.RENDER_ATTACHMENT;
    this._msaaTex = this.device.createTexture(
      { size, format: this.colorFormat, sampleCount: SAMPLE_COUNT, usage });
    this._depthTex = this.device.createTexture(
      { size, format: DEPTH_FORMAT, sampleCount: SAMPLE_COUNT, usage });
    this._targetSize = [width, height];
  }

  // ------------------------------------------------------------------ scene

  /** Drop every model and free its buffers -- called on each scene_init. */
  clear() {
    this.models.forEach((model) => {
      model.buffers.forEach((buffer) => buffer.destroy());
    });
    this.models.clear();
  }

  /** Drop one model -- the page half of a scene_delta removal. */
  remove(id) {
    const model = this.models.get(id);
    if (!model) return;
    model.buffers.forEach((buffer) => buffer.destroy());
    this.models.delete(id);
  }

  /** Add one drawable, dispatching on the kind the server tagged it with. */
  add(payload) {
    if (payload.kind === 'pcd') this._addPointCloud(payload);
    else this._addMesh(payload);
  }

  _addMesh({ id, vertices, faces, normals, rgba }) {
    if (!faces || faces.length === 0) return;
    const device = this.device;
    const instanceUsage = GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST;

    const vertexBuf = createBufferWithData(
      device, interleave3x2(vertices, normals), GPUBufferUsage.VERTEX);
    // faces already arrives as a Uint32Array view on the socket buffer
    const indexBuf = createBufferWithData(device, faces, GPUBufferUsage.INDEX);
    const tfBuf = device.createBuffer({ size: 64, usage: instanceUsage });
    const rgbaBuf = createBufferWithData(
      device, new Float32Array(rgba), instanceUsage);

    this.models.set(id, {
      kind: 'mesh',
      buffers: [vertexBuf, indexBuf, tfBuf, rgbaBuf],
      vertexBuf,
      indexBuf,
      rgbaBuf,
      tfBuf,
      indexCount: faces.length,
      // render.py splits the same way: alpha < 0.999 goes to the transparent
      // pipeline, and only the solid group gets an outline pass
      opaque: rgba[3] >= 0.999,
      ...this._freshTransform(tfBuf),
    });
  }

  _addPointCloud({ id, points, colors }) {
    if (!points || points.length === 0) return;
    const device = this.device;

    const instanceBuf = createBufferWithData(
      device, interleave3x2(points, colors), GPUBufferUsage.VERTEX);
    const modelBuf = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.models.set(id, {
      kind: 'pcd',
      buffers: [instanceBuf, modelBuf],
      instanceBuf,
      tfBuf: modelBuf,
      modelBindGroup: device.createBindGroup({
        layout: this.modelLayout,
        entries: [{ binding: 0, resource: { buffer: modelBuf } }],
      }),
      count: points.length / 3,
      ...this._freshTransform(modelBuf),
    });
  }

  /**
   * Identity until the first scene_update lands, so nothing is ever drawn
   * with an uninitialised (all-zero, hence degenerate) model matrix.
   */
  _freshTransform(buffer) {
    const tf = new Float32Array(IDENTITY);
    this.device.queue.writeBuffer(buffer, 0, tf);
    return { tf, tfDirty: false };
  }

  setTransform(id, matrix) {
    const model = this.models.get(id);
    if (!model) return;
    model.tf.set(matrix);   // already column-major on the wire; see math.js
    model.tfDirty = true;
  }

  // ------------------------------------------------------------------- draw

  render(camera) {
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const height = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== width) this.canvas.width = width;
    if (this.canvas.height !== height) this.canvas.height = height;
    this._ensureTargets(width, height);

    // --- prepare: every upload happens before the pass opens
    camera.writeView(this._view);
    camera.writeProj(this._proj, width / height);
    this._globals.set(camera.pos, 32);
    this._globals[36] = width;
    this._globals[37] = height;
    this._globals[38] = POINT_SIZE;
    this.device.queue.writeBuffer(this.globalsBuf, 0, this._globals);

    const solid = [];
    const transparent = [];
    const pcd = [];
    this.models.forEach((model) => {
      if (model.tfDirty) {
        this.device.queue.writeBuffer(model.tfBuf, 0, model.tf);
        model.tfDirty = false;
      }
      if (model.kind === 'pcd') pcd.push(model);
      else (model.opaque ? solid : transparent).push(model);
    });
    // _prepare_transparent: back-to-front by squared distance to the eye.
    // Column-major puts the translation at 12..14.
    if (transparent.length > 1) {
      const eye = camera.pos;
      transparent.forEach((model) => {
        const dx = model.tf[12] - eye[0];
        const dy = model.tf[13] - eye[1];
        const dz = model.tf[14] - eye[2];
        model._depth = dx * dx + dy * dy + dz * dz;
      });
      transparent.sort((a, b) => b._depth - a._depth);
    }

    // --- draw
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this._msaaTex.createView(),
        resolveTarget: this.context.getCurrentTexture()
          .createView({ format: this.colorFormat }),
        clearValue: this.clearValue,
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this._depthTex.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    pass.setBindGroup(0, this.globalsBindGroup);

    const drawMeshes = (list) => list.forEach((model) => {
      pass.setVertexBuffer(0, model.vertexBuf);
      pass.setVertexBuffer(1, model.tfBuf);
      pass.setVertexBuffer(2, model.rgbaBuf);
      pass.setIndexBuffer(model.indexBuf, 'uint32');
      pass.drawIndexed(model.indexCount, 1);
    });

    if (solid.length) {
      pass.setPipeline(this.outlinePipeline);
      drawMeshes(solid);
      pass.setPipeline(this.solidPipeline);
      drawMeshes(solid);
    }
    if (transparent.length) {
      pass.setPipeline(this.transparentPipeline);
      drawMeshes(transparent);
    }
    if (pcd.length) {
      pass.setPipeline(this.pcdPipeline);
      pass.setVertexBuffer(0, this.quadBuf);
      pcd.forEach((model) => {
        pass.setBindGroup(1, model.modelBindGroup);
        pass.setVertexBuffer(1, model.instanceBuf);
        pass.draw(6, model.count);
      });
    }

    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}
