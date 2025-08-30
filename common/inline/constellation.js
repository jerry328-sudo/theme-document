/*
 * Constellation Particles Background
 * - Physics upgraded to Newtonian N-body with softening
 * - Optional Paczyński–Wiita pseudo-Newton potential near heavy masses
 * - Velocity-Verlet integrator; inelastic merge on close encounters
 * - HiDPI, resize-aware, visibility-aware
 * - Only runs in dark mode (html.dark); hidden in light mode
*/
(function () {
  if (typeof window === 'undefined' || typeof document === 'undefined') return;

  const root = document.documentElement;
  const ID = 'constellation';
  let canvas = document.getElementById(ID);
  if (!canvas) {
    canvas = document.createElement('canvas');
    canvas.id = ID;
    document.body.prepend(canvas);
  }
  const ctx = canvas.getContext('2d', { alpha: true });

  // Config
  const CONFIG = {
    density: 0.00005,     // particles per pixel^2
    maxCount: 260,
    minCount: 80,
    dotRadius: [1.0, 2.0],
    // 可视速度限制（仅用于绘制插值/尾巴稳定，不改变物理速度）
    visualSpeedMax: 6,
    // 引力与相对论近似参数（屏幕单位）
    alphaExponent: 3,   // m ~ s^alpha
    massDensity: 1,     // ρ：将像素半径映射为质量
    usePW: true,        // 近重体使用 Paczyński–Wiita 近似
    kappaPW: 10,        // 触发 PW 的 r < kappa * r_s 阈值
    mergeEta: 0.8,      // 并合判据系数：r_ij < eta*(s_i+s_j)
    softeningScale: 0.6, // ε ≈ scale * median(s)
    maxMergesPerFrame: 8,
    // 时间步长与常数标定
    dtScale: 1,         // 基于 60fps 的归一化步长缩放
    circVelFactor: 0.5, // v_circ ≈ factor * visualSpeedMax for G 标定
    // 相对论光速（屏幕单位）
    lightSpeedFactor: 15, // c* ≈ factor * visualSpeedMax
    // 鼠标吸引弱外力（相对引力应很小）
    drawMouseLinks: true, // 是否绘制鼠标连线
    linkDistance: 120, // 连线最大距离
    linkWidth: 0.8, // 连线宽度
    linkOpacity: 0.7, // 最大不透明度
    hoverRadius: 160, // 鼠标引力作用半径（决定衰减范围）
    attractStrength: 0.1, // 鼠标吸引强度（加速度），远弱于引力
    dotColor: 'rgba(255,255,255,1)',
    lineColor: [255, 255, 255], 
    twinkle: true, // 点闪烁
    twinkleAmplitude: 0.35, // 闪烁幅度
    // 线段型尾巴（已默认关闭，保留作为备选）
    trailEnabled: false,
    trailFadeTime: 700,
    trailMaxPoints: 48,
    trailMinDist: 3,
    trailWidth: 7,
    trailOpacity: 0.75,
    // 粒子型彗星尾巴（默认开启）
    cometEnabled: true,
    cometSpawnDist: 4,         // 鼠标移动每隔多少像素生成一个粒子
    cometLife: 800,            // 单个尾巴粒子寿命（ms）
    cometLifeJitter: 250,      // 寿命抖动（±ms）
    cometSize: [0.6, 1.6],     // 尾巴粒子半径范围
    cometSpeedScale: 0.12,     // 初速度与移动距离的比例（越大越拖尾）
    cometJitter: 0.25,         // 初速度随机抖动比例
    cometDamping: 0.04,        // 速度阻尼
    cometOpacity: 0.9,         // 最高不透明度
    cometMaxCount: 500         // 尾巴粒子最大数量（防止过多导致卡顿）
    ,
    // WebGPU 开关
    enableWebGPU: true
  };

  // 常量与可复用结构（避免每帧创建临时对象）
  const NEIGHBOR_OFFS = [ [0,0], [1,0], [0,1], [1,1], [1,-1] ];
  const LINK_BIN_COUNT = 16; // 连线透明度量化桶数（批量描边）
  const linkBins = Array.from({ length: LINK_BIN_COUNT }, () => []); // 每桶存放 [x1,y1,x2,y2,...]
  const BASE_STROKE = `rgb(${CONFIG.lineColor[0]},${CONFIG.lineColor[1]},${CONFIG.lineColor[2]})`;

  let isDark = root.classList.contains('dark');
  let dpr = Math.min(window.devicePixelRatio || 1, 2);
  let targetCount = 0;
  const particles = [];
  // 物理状态（避免每帧重新分配）
  let ax = [], ay = [], axNext = [], ayNext = [];
  let haveAccel = false;
  let EPS = 1.0;   // 软化长度 ε（像素）
  let Gstar = 0.002; // 引力常数（屏幕单位）
  let Cstar = 20;    // 光速（屏幕像素/帧单位）
  let PW_DELTA = 0.2; // PW 钳位相对 ε 的比例，实际值在 resize 后设
  const rand = (a, b) => Math.random() * (b - a) + a;

  // WebGPU 状态
  const webgpu = {
    available: false,
    device: null,
    pipeline: null,
    bindGroup: null,
    uniformBuffer: null,
    posBuffer: null,
    massBuffer: null,
    accBuffer: null,
    workgroupSize: 64,
    capacity: 0,
    posF32: null,
    massF32: null,
    accF32: null
  };

  async function initWebGPU() {
    try {
      if (!CONFIG.enableWebGPU || !('gpu' in navigator)) return;
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return;
      const device = await adapter.requestDevice();
      webgpu.device = device;
      webgpu.available = true;
      webgpu.capacity = CONFIG.maxCount;
      const cap = webgpu.capacity;
      // Typed arrays for IO
      webgpu.posF32 = new Float32Array(cap * 2);
      webgpu.massF32 = new Float32Array(cap);
      webgpu.accF32 = new Float32Array(cap * 2);

      // WGSL compute shader (tiled N^2)
      const wgsl = `
        struct Params {
          N: u32,
          usePW: u32,
          padA: vec2<u32>,
          eps2: f32,
          G: f32,
          C: f32,
          kappa: f32,
        };
        @group(0) @binding(0) var<storage, read> pos: array<vec2<f32>>;
        @group(0) @binding(1) var<storage, read> mass: array<f32>;
        @group(0) @binding(2) var<storage, read_write> acc: array<vec2<f32>>;
        @group(0) @binding(3) var<uniform> params: Params;

        var<workgroup> tPos: array<vec2<f32>, 64>;
        var<workgroup> tMass: array<f32, 64>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid : vec3<u32>,
                @builtin(local_invocation_id) lid : vec3<u32>) {
          let i = gid.x;
          if (i >= params.N) { return; }
          var ai = vec2<f32>(0.0, 0.0);
          let N = params.N;
          var base: u32 = 0u;
          loop {
            if (base >= N) { break; }
            let j = base + lid.x;
            if (j < N) {
              tPos[lid.x] = pos[j];
              tMass[lid.x] = mass[j];
            }
            workgroupBarrier();
            let maxJ = min(base + 64u, N);
            var jj: u32 = base;
            loop {
              if (jj >= maxJ) { break; }
              if (jj != i) {
                let d = tPos[jj - base] - pos[i];
                let r2 = d.x*d.x + d.y*d.y;
                let r = sqrt(max(r2, 1e-12));
                let ux = d.x / r;
                let uy = d.y / r;
                var axv: f32;
                var ayv: f32;
                if (params.usePW == 1u) {
                  let rs = 2.0 * params.G * tMass[jj - base] / (params.C * params.C);
                  if (r < params.kappa * rs) {
                    let reff = max(0.2 * sqrt(params.eps2), r - rs);
                    let mag = params.G * tMass[jj - base] / (reff * reff);
                    axv = ux * mag; ayv = uy * mag;
                  } else {
                    let inv = inverseSqrt(r2 + params.eps2);
                    let inv3 = inv * inv * inv;
                    let mag = params.G * tMass[jj - base] * inv3;
                    axv = d.x * mag; ayv = d.y * mag;
                  }
                } else {
                  let inv = inverseSqrt(r2 + params.eps2);
                  let inv3 = inv * inv * inv;
                  let mag = params.G * tMass[jj - base] * inv3;
                  axv = d.x * mag; ayv = d.y * mag;
                }
                ai.x = ai.x + axv;
                ai.y = ai.y + ayv;
              }
              jj = jj + 1u;
            }
            workgroupBarrier();
            base = base + 64u;
          }
          acc[i] = ai;
        }
      `;
      const module = device.createShaderModule({ code: wgsl });
      const pipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module, entryPoint: 'main' }
      });
      webgpu.pipeline = pipeline;

      // Buffers
      const posBuffer = device.createBuffer({ size: cap * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const massBuffer = device.createBuffer({ size: cap * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const accBuffer = device.createBuffer({ size: cap * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const uniformBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      webgpu.posBuffer = posBuffer;
      webgpu.massBuffer = massBuffer;
      webgpu.accBuffer = accBuffer;
      webgpu.uniformBuffer = uniformBuffer;

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: posBuffer } },
          { binding: 1, resource: { buffer: massBuffer } },
          { binding: 2, resource: { buffer: accBuffer } },
          { binding: 3, resource: { buffer: uniformBuffer } },
        ]
      });
      webgpu.bindGroup = bindGroup;
    } catch (e) {
      // 失败则静默降级到 CPU
      webgpu.available = false;
      console.warn('WebGPU init failed; falling back to CPU:', e);
    }
  }

  function updateUniformsForGPU(N) {
    if (!webgpu.available) return;
    const ub = new ArrayBuffer(32);
    const u32 = new Uint32Array(ub);
    const f32 = new Float32Array(ub);
    u32[0] = N >>> 0;
    u32[1] = CONFIG.usePW ? 1 : 0;
    // u32[2], u32[3] are padding
    f32[4] = EPS * EPS;
    f32[5] = Gstar;
    f32[6] = Cstar;
    f32[7] = CONFIG.kappaPW;
    webgpu.device.queue.writeBuffer(webgpu.uniformBuffer, 0, ub);
  }

  async function computeAccelerationsGPU(outAx, outAy) {
    const N = particles.length;
    if (!webgpu.available || N === 0) return false;
    // Pack current positions and masses
    const pos = webgpu.posF32; const mass = webgpu.massF32; const acc = webgpu.accF32;
    for (let i = 0; i < N; i++) {
      const p = particles[i];
      pos[2*i] = p.x; pos[2*i+1] = p.y; mass[i] = p.m;
    }
    const dev = webgpu.device;
    dev.queue.writeBuffer(webgpu.posBuffer, 0, pos, 0, N * 2);
    dev.queue.writeBuffer(webgpu.massBuffer, 0, mass, 0, N);
    updateUniformsForGPU(N);

    const encoder = dev.createCommandEncoder();
    const cpass = encoder.beginComputePass();
    cpass.setPipeline(webgpu.pipeline);
    cpass.setBindGroup(0, webgpu.bindGroup);
    const groups = Math.ceil(N / webgpu.workgroupSize);
    cpass.dispatchWorkgroups(groups);
    cpass.end();
    dev.queue.submit([encoder.finish()]);

    // 读取加速度结果
    // 优先使用 queue.readBuffer（若存在），否则回退 staging buffer + map
    const readTo = webgpu.accF32.subarray(0, N * 2);
    if (dev.queue.readBuffer) {
      await dev.queue.readBuffer(webgpu.accBuffer, 0, readTo);
    } else {
      const staging = dev.createBuffer({ size: N * 2 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc2 = dev.createCommandEncoder();
      enc2.copyBufferToBuffer(webgpu.accBuffer, 0, staging, 0, N * 2 * 4);
      dev.queue.submit([enc2.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      readTo.set(new Float32Array(staging.getMappedRange()));
      staging.unmap();
      staging.destroy();
    }
    for (let i = 0; i < N; i++) { outAx[i] = readTo[2*i]; outAy[i] = readTo[2*i+1]; }
    // 叠加鼠标弱外力（CPU 侧补充）
    if (mouse.active) {
      const R = CONFIG.hoverRadius || CONFIG.linkDistance;
      for (let i = 0; i < N; i++) {
        const p = particles[i];
        const dxm = mouse.x - p.x; const dym = mouse.y - p.y;
        const rm = Math.hypot(dxm, dym) + 1e-12;
        if (rm < R) {
          const t = 1 - rm / R; // 0..1
          const falloff = t * t; // 快速二次衰减
          outAx[i] += (dxm / rm) * CONFIG.attractStrength * falloff;
          outAy[i] += (dym / rm) * CONFIG.attractStrength * falloff;
        }
      }
    }
    return true;
  }

  function computeTargetCount() {
    const area = window.innerWidth * window.innerHeight;
    targetCount = Math.max(CONFIG.minCount, Math.min(CONFIG.maxCount, Math.round(area * CONFIG.density)));
  }

  function resize() {
    const { innerWidth: w, innerHeight: h } = window;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.floor(w * dpr));
    canvas.height = Math.max(1, Math.floor(h * dpr));
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    computeTargetCount();
    ensureCount();
    // 更新软化长度与常数标定
    updateSofteningAndConstants();
    haveAccel = false; // 尺寸变更需要重算加速度
  }

  function massFromRadius(r) {
    return CONFIG.massDensity * Math.pow(Math.max(0.1, r), CONFIG.alphaExponent);
  }

  function spawnParticle() {
    const r = rand(CONFIG.dotRadius[0], CONFIG.dotRadius[1]);
    particles.push({
      x: rand(0, window.innerWidth),
      y: rand(0, window.innerHeight),
      vx: rand(-CONFIG.visualSpeedMax * 0.2, CONFIG.visualSpeedMax * 0.2),
      vy: rand(-CONFIG.visualSpeedMax * 0.2, CONFIG.visualSpeedMax * 0.2),
      r,
      m: massFromRadius(r),
      t: Math.random() * Math.PI * 2
    });
  }

  function ensureCount() {
    while (particles.length < targetCount) spawnParticle();
    if (particles.length > targetCount) particles.length = targetCount;
    // 确保加速度数组与粒子数量匹配
    const n = particles.length;
    ax.length = ay.length = axNext.length = ayNext.length = n;
    for (let i = 0; i < n; i++) { ax[i] = ax[i] || 0; ay[i] = ay[i] || 0; axNext[i] = axNext[i] || 0; ayNext[i] = ayNext[i] || 0; }
  }

  // Mouse / touch
  const mouse = { x: 0, y: 0, active: false, dragging: false };
  const mouseTrail = []; // 线段尾巴 {x,y,t}
  const cometParticles = []; // 粒子尾巴 {x,y,vx,vy,life,lifeMax,r}
  const lastMouse = { x: 0, y: 0, has: false };
  function clearTrail() { mouseTrail.length = 0; }
  function clearComet() { cometParticles.length = 0; }
  function deactivateMouse() {
    mouse.active = false;
    mouse.dragging = false;
    // 将坐标移到极远处，避免边缘残留吸引/连线
    mouse.x = -999999;
    mouse.y = -999999;
    clearTrail();
    // 粒子尾巴不立即清空，保留自然淡出；但重置参考点
    lastMouse.has = false;
  }
  function setMouse(e) {
    const rect = canvas.getBoundingClientRect();
    mouse.x = e.clientX - rect.left;
    mouse.y = e.clientY - rect.top;
    mouse.active = true;
    if (mouse.dragging && CONFIG.trailEnabled) addTrailPoint(mouse.x, mouse.y);
    if (CONFIG.cometEnabled) addCometFromMove(mouse.x, mouse.y);
  }
  function addTrailPoint(x, y) {
    const now = performance.now();
    const n = mouseTrail.length;
    if (n > 0) {
      const lp = mouseTrail[n - 1];
      const dx = x - lp.x, dy = y - lp.y;
      if (dx * dx + dy * dy < CONFIG.trailMinDist * CONFIG.trailMinDist) return;
    }
    mouseTrail.push({ x, y, t: now });
    if (mouseTrail.length > CONFIG.trailMaxPoints) mouseTrail.shift();
  }
  window.addEventListener('mousemove', setMouse, { passive: true });
  window.addEventListener('mouseenter', () => (mouse.active = true), { passive: true });
  window.addEventListener('mouseleave', deactivateMouse, { passive: true });
  // 更可靠的离开检测：当 relatedTarget 为空时，表示鼠标离开了页面
  window.addEventListener('mouseout', (e) => { if (!e.relatedTarget && !e.toElement) deactivateMouse(); }, { passive: true });
  document.addEventListener('mouseleave', deactivateMouse, { passive: true });
  // 指针事件，进一步增强跨浏览器一致性
  document.addEventListener('pointerleave', deactivateMouse, { passive: true });
  window.addEventListener('pointerout', (e) => { if (!e.relatedTarget) deactivateMouse(); }, { passive: true });
  document.addEventListener('pointerdown', (e) => { setMouse(e); mouse.dragging = true; addTrailPoint(mouse.x, mouse.y); }, { passive: true });
  document.addEventListener('pointerup', () => { mouse.dragging = false; }, { passive: true });
  // 兼容不支持 Pointer Events 的浏览器
  document.addEventListener('mousedown', (e) => { setMouse(e); mouse.dragging = true; addTrailPoint(mouse.x, mouse.y); }, { passive: true });
  document.addEventListener('mouseup', () => { mouse.dragging = false; }, { passive: true });
  window.addEventListener('blur', deactivateMouse);
  window.addEventListener('touchstart', (e) => { const t = e.touches[0]; if (t) { mouse.dragging = true; setMouse(t); addTrailPoint(mouse.x, mouse.y); } }, { passive: true });
  window.addEventListener('touchmove', (e) => { const t = e.touches[0]; if (t) setMouse(t); }, { passive: true });
  window.addEventListener('touchend', () => { mouse.dragging = false; deactivateMouse(); });
  window.addEventListener('touchcancel', () => { mouse.dragging = false; deactivateMouse(); });

  // Comet particle helpers
  function randRange(a, b) { return Math.random() * (b - a) + a; }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function addCometFromMove(x, y) {
    const now = performance.now();
    if (!lastMouse.has) { lastMouse.x = x; lastMouse.y = y; lastMouse.has = true; return; }
    const dx = x - lastMouse.x; const dy = y - lastMouse.y; const dist = Math.hypot(dx, dy);
    if (dist <= 0.0001) return;
    const steps = Math.max(1, Math.floor(dist / CONFIG.cometSpawnDist));
    const ux = dx / dist, uy = dy / dist;
    for (let i = 1; i <= steps; i++) {
      const px = lerp(lastMouse.x, x, i / steps);
      const py = lerp(lastMouse.y, y, i / steps);
      // 速度与移动方向相反，使粒子跟随拖尾
      const baseV = Math.min(60, dist) * CONFIG.cometSpeedScale; // 基于本次位移，限制上限
      const jx = (Math.random() - 0.5) * 2 * CONFIG.cometJitter * baseV;
      const jy = (Math.random() - 0.5) * 2 * CONFIG.cometJitter * baseV;
      const vx = -ux * baseV + jx;
      const vy = -uy * baseV + jy;
      const lifeMax = Math.max(200, CONFIG.cometLife + randRange(-CONFIG.cometLifeJitter, CONFIG.cometLifeJitter));
      const r = randRange(CONFIG.cometSize[0], CONFIG.cometSize[1]);
      cometParticles.push({ x: px, y: py, vx, vy, life: lifeMax, lifeMax, r });
    }
    // 控制上限
    if (cometParticles.length > CONFIG.cometMaxCount) {
      const overflow = cometParticles.length - CONFIG.cometMaxCount;
      cometParticles.splice(0, overflow);
    }
    lastMouse.x = x; lastMouse.y = y; lastMouse.has = true;
  }

  // Grid for neighborhood search（复用内存，减少 GC）
  let cellSize = CONFIG.linkDistance;
  let grid = [];
  let gridCols = 0, gridRows = 0, gridCellSize = 0;
  function buildGrid() {
    cellSize = CONFIG.linkDistance;
    const cols = Math.ceil(window.innerWidth / cellSize);
    const rows = Math.ceil(window.innerHeight / cellSize);
    if (cols !== gridCols || rows !== gridRows || gridCellSize !== cellSize || grid.length !== cols * rows) {
      gridCols = cols; gridRows = rows; gridCellSize = cellSize;
      grid = new Array(cols * rows);
      for (let i = 0; i < grid.length; i++) grid[i] = [];
    } else {
      for (let i = 0; i < grid.length; i++) grid[i].length = 0;
    }
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      let cx = (p.x / cellSize) | 0; let cy = (p.y / cellSize) | 0;
      if (cx < 0) cx = 0; else if (cx >= cols) cx = cols - 1;
      if (cy < 0) cy = 0; else if (cy >= rows) cy = rows - 1;
      grid[cy * cols + cx].push(i);
    }
    return { grid, cols, rows };
  }

  // === 物理相关辅助 ===
  function totalMass() {
    let s = 0; for (let i = 0; i < particles.length; i++) s += particles[i].m; return s;
  }

  function medianRadius() {
    if (particles.length === 0) return (CONFIG.dotRadius[0] + CONFIG.dotRadius[1]) * 0.5;
    const arr = particles.map(p => p.r).sort((a,b)=>a-b);
    const mid = arr.length >> 1; return arr.length % 2 ? arr[mid] : 0.5*(arr[mid-1]+arr[mid]);
  }

  function updateSofteningAndConstants() {
    const med = medianRadius();
    EPS = Math.max(0.2, CONFIG.softeningScale * med);
    PW_DELTA = 0.2 * EPS;
    // 标定 G：v_circ^2 ≈ G M / R -> G ≈ v^2 R / M
    const vCirc = Math.max(0.1, CONFIG.circVelFactor * CONFIG.visualSpeedMax);
    const R = Math.min(window.innerWidth, window.innerHeight) * 0.25;
    const M = Math.max(1e-6, totalMass());
    Gstar = (vCirc * vCirc) * (R / M);
    Cstar = Math.max(vCirc * 10, CONFIG.lightSpeedFactor * CONFIG.visualSpeedMax);
  }

  function computeAccelerationsCPU(outAx, outAy) {
    const n = particles.length;
    for (let i = 0; i < n; i++) { outAx[i] = 0; outAy[i] = 0; }
    // 朴素 O(N^2) 求和（N<=260 可接受）
    for (let i = 0; i < n; i++) {
      const pi = particles[i];
      for (let j = i + 1; j < n; j++) {
        const pj = particles[j];
        let dx = pj.x - pi.x; let dy = pj.y - pi.y;
        const r2 = dx*dx + dy*dy;
        const r = Math.sqrt(r2 + 1e-12);
        const ux = dx / r, uy = dy / r;
        // i <- j
        let ax_ij = 0, ay_ij = 0;
        if (CONFIG.usePW) {
          const rsj = (2 * Gstar * pj.m) / (Cstar * Cstar);
          if (r < CONFIG.kappaPW * rsj) {
            const reff = Math.max(PW_DELTA, r - rsj);
            const mag = (Gstar * pj.m) / (reff * reff);
            ax_ij = ux * mag; ay_ij = uy * mag;
          } else {
            const inv = 1 / Math.pow(r2 + EPS*EPS, 1.5);
            ax_ij = dx * (Gstar * pj.m * inv);
            ay_ij = dy * (Gstar * pj.m * inv);
          }
        } else {
          const inv = 1 / Math.pow(r2 + EPS*EPS, 1.5);
          ax_ij = dx * (Gstar * pj.m * inv);
          ay_ij = dy * (Gstar * pj.m * inv);
        }
        outAx[i] += ax_ij; outAy[i] += ay_ij;

        // j <- i（对称）
        if (CONFIG.usePW) {
          const rsi = (2 * Gstar * pi.m) / (Cstar * Cstar);
          if (r < CONFIG.kappaPW * rsi) {
            const reff = Math.max(PW_DELTA, r - rsi);
            const mag = (Gstar * pi.m) / (reff * reff);
            // 方向相反
            outAx[j] -= ux * mag; outAy[j] -= uy * mag;
          } else {
            const inv = 1 / Math.pow(r2 + EPS*EPS, 1.5);
            outAx[j] -= dx * (Gstar * pi.m * inv);
            outAy[j] -= dy * (Gstar * pi.m * inv);
          }
        } else {
          const inv = 1 / Math.pow(r2 + EPS*EPS, 1.5);
          outAx[j] -= dx * (Gstar * pi.m * inv);
          outAy[j] -= dy * (Gstar * pi.m * inv);
        }
      }
      // 鼠标吸引（弱外力，局部快速衰减）
      if (mouse.active) {
        const dxm = mouse.x - pi.x; const dym = mouse.y - pi.y;
        const rm = Math.hypot(dxm, dym) + 1e-12;
        const R = CONFIG.hoverRadius || CONFIG.linkDistance;
        if (rm < R) {
          const t = 1 - rm / R; // 0..1
          const falloff = t * t; // 二次衰减，半径外无影响
          outAx[i] += (dxm / rm) * CONFIG.attractStrength * falloff;
          outAy[i] += (dym / rm) * CONFIG.attractStrength * falloff;
        }
      }
    }
  }

  async function computeAccelerations(outAx, outAy) {
    if (webgpu.available) {
      try {
        const ok = await computeAccelerationsGPU(outAx, outAy);
        if (ok) return;
      } catch (e) {
        console.warn('WebGPU compute failed, fallback to CPU:', e);
        webgpu.available = false;
      }
    }
    computeAccelerationsCPU(outAx, outAy);
  }

  function mergePairs(limit = CONFIG.maxMergesPerFrame) {
    const n = particles.length; if (n <= 1) return;
    const toRemove = new Set();
    let merges = 0;
    for (let i = 0; i < n; i++) {
      if (toRemove.has(i)) continue;
      const pi = particles[i];
      for (let j = i + 1; j < n; j++) {
        if (toRemove.has(j)) continue;
        const pj = particles[j];
        const dx = pj.x - pi.x; const dy = pj.y - pi.y;
        const r = Math.hypot(dx, dy);
        if (r < CONFIG.mergeEta * (pi.r + pj.r)) {
          // 并合：动量守恒 + 质心
          const mnew = pi.m + pj.m;
          const vx = (pi.vx * pi.m + pj.vx * pj.m) / mnew;
          const vy = (pi.vy * pi.m + pj.vy * pj.m) / mnew;
          const x = (pi.x * pi.m + pj.x * pj.m) / mnew;
          const y = (pi.y * pi.m + pj.y * pj.m) / mnew;
          const rnew = Math.pow(Math.max(1e-9, mnew / CONFIG.massDensity), 1 / CONFIG.alphaExponent);
          // 写回 i，标记 j 删除
          pi.x = x; pi.y = y; pi.vx = vx; pi.vy = vy; pi.m = mnew; pi.r = rnew;
          toRemove.add(j);
          merges++;
          if (merges >= limit) break;
        }
      }
      if (merges >= limit) break;
    }
    if (toRemove.size > 0) {
      const newList = [];
      for (let k = 0; k < particles.length; k++) if (!toRemove.has(k)) newList.push(particles[k]);
      particles.length = 0; particles.push(...newList);
      ensureCount();
      updateSofteningAndConstants();
      haveAccel = false; // 粒子发生变化，下帧重算加速度
    }
  }

  // Animation loop
  let raf = 0;
  let lastTime = performance.now();
  async function loop(now) {
    // 基于 60fps 归一时间步长
    const dt = CONFIG.dtScale * Math.max(0.5, Math.min(2.0, (now - lastTime) / (1000 / 60)));
    lastTime = now;

    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    // 物理：Velocity-Verlet 积分（经典区），PW 近似在 computeAccelerations 内处理
    if (!haveAccel) { await computeAccelerations(ax, ay); haveAccel = true; }
    const n = particles.length;
    // 位置推进（x += v dt + 0.5 a dt^2）
    const dt2 = dt * dt;
    for (let i = 0; i < n; i++) {
      const p = particles[i];
      p.x += p.vx * dt + 0.5 * ax[i] * dt2;
      p.y += p.vy * dt + 0.5 * ay[i] * dt2;
      // 边界条件：反弹
      if (p.x < 0) { p.x = 0; p.vx *= -1; }
      else if (p.x > window.innerWidth) { p.x = window.innerWidth; p.vx *= -1; }
      if (p.y < 0) { p.y = 0; p.vy *= -1; }
      else if (p.y > window.innerHeight) { p.y = window.innerHeight; p.vy *= -1; }
    }
    // 新位置上的加速度
    await computeAccelerations(axNext, ayNext);
    // 速度推进（v += 0.5 (a + aNext) dt），并施加可视限速
    const vmax = CONFIG.visualSpeedMax;
    for (let i = 0; i < n; i++) {
      const p = particles[i];
      p.vx += 0.5 * (ax[i] + axNext[i]) * dt;
      p.vy += 0.5 * (ay[i] + ayNext[i]) * dt;
      // 可视限速（不修改物理能量，只为视觉稳定）
      const sp = Math.hypot(p.vx, p.vy);
      if (sp > vmax) { const k = vmax / sp; p.vx *= k; p.vy *= k; }
      ax[i] = axNext[i]; ay[i] = ayNext[i];
    }
    // 碰撞并合（动量守恒）
    mergePairs();

    // Linking using grid（复用一次网格，按透明度分桶后批量描边）
    const { grid: grid2, cols, rows } = buildGrid();
    const maxD = CONFIG.linkDistance; const maxD2 = maxD * maxD;
    for (let b = 0; b < LINK_BIN_COUNT; b++) linkBins[b].length = 0;

    function addLink(a, b, opacityScale) {
      const dx = a.x - b.x; const dy = a.y - b.y; const d2 = dx * dx + dy * dy; if (d2 > maxD2) return;
      const d = Math.sqrt(d2);
      const alpha = (1 - d / maxD) * CONFIG.linkOpacity * opacityScale;
      if (alpha <= 0) return;
      const bi = Math.min(LINK_BIN_COUNT - 1, ((alpha / CONFIG.linkOpacity) * (LINK_BIN_COUNT - 1)) | 0);
      const segs = linkBins[bi];
      segs.push(a.x, a.y, b.x, b.y);
    }

    for (let cy = 0; cy < rows; cy++) {
      for (let cx = 0; cx < cols; cx++) {
        const bucket = grid2[cy * cols + cx]; if (!bucket) continue;
        // 同格内两两组合
        for (let ii = 0; ii < bucket.length; ii++) {
          const ai = bucket[ii]; const a = particles[ai];
          for (let jj = ii + 1; jj < bucket.length; jj++) addLink(a, particles[bucket[jj]], 1);
        }
        // 邻域格
        for (let k = 1; k < NEIGHBOR_OFFS.length; k++) { // 跳过 [0,0]
          const nx = cx + NEIGHBOR_OFFS[k][0], ny = cy + NEIGHBOR_OFFS[k][1];
          if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) continue;
          const nb = grid2[ny * cols + nx]; if (!nb) continue;
          for (let ii = 0; ii < bucket.length; ii++) {
            const a = particles[bucket[ii]];
            for (let jj = 0; jj < nb.length; jj++) addLink(a, particles[nb[jj]], 1);
          }
        }
      }
    }

    // 鼠标连线也加入同一分桶，统一批量描边
    if (mouse.active && CONFIG.drawMouseLinks) {
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        const dx = p.x - mouse.x; const dy = p.y - mouse.y; const d2 = dx * dx + dy * dy;
        if (d2 <= maxD2) addLink(p, mouse, 0.9);
      }
    }

    // 批量描边：每个透明度桶一次 stroke
    ctx.strokeStyle = BASE_STROKE;
    ctx.lineWidth = CONFIG.linkWidth;
    for (let b = 0; b < LINK_BIN_COUNT; b++) {
      const segs = linkBins[b];
      if (segs.length === 0) continue;
      const alpha = ((b + 0.5) / LINK_BIN_COUNT) * CONFIG.linkOpacity;
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      for (let i = 0; i < segs.length; i += 4) { ctx.moveTo(segs[i], segs[i+1]); ctx.lineTo(segs[i+2], segs[i+3]); }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Draw particles
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i]; let alpha = 1;
      if (CONFIG.twinkle) { p.t += 0.02; alpha = 1 - (Math.sin(p.t) * 0.5 + 0.5) * CONFIG.twinkleAmplitude; }
      const dotA = (0.75 + 0.25 * alpha).toFixed(3);
      ctx.fillStyle = `rgba(255,255,255,${dotA})`;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
    }

    // Mouse comet trail (line-based, optional)
    if (CONFIG.trailEnabled && mouseTrail.length > 1) {
      // 移除过期点
      const expire = now - CONFIG.trailFadeTime;
      let startIdx = -1;
      for (let i = 0; i < mouseTrail.length; i++) { if (mouseTrail[i].t >= expire) { startIdx = i; break; } }
      if (startIdx === -1) {
        // 全部过期
        clearTrail();
      } else if (startIdx > 0) {
        mouseTrail.splice(0, startIdx);
      }

      // 绘制分段线，宽度与透明度随时间与顺序衰减
      const n = mouseTrail.length;
      if (n > 1) {
        const rgb = CONFIG.lineColor || [255, 255, 255];
        const prevCap = ctx.lineCap; ctx.lineCap = 'round';
        for (let i = 1; i < n; i++) {
          const p0 = mouseTrail[i - 1];
          const p1 = mouseTrail[i];
          const age = Math.max(0, now - p1.t);
          const k = 1 - Math.min(1, age / CONFIG.trailFadeTime); // 0..1
          const ease = k * k; // quadratic ease-out
          const alpha = Math.max(0, ease * CONFIG.trailOpacity);
          const width = Math.max(1, CONFIG.trailWidth * (0.3 + 0.7 * ease));
          ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha.toFixed(3)})`;
          ctx.lineWidth = width;
          ctx.beginPath();
          ctx.moveTo(p0.x, p0.y);
          ctx.lineTo(p1.x, p1.y);
          ctx.stroke();
        }
        ctx.lineCap = prevCap;
      }
    }

    // Comet particles (glowing, without press)
    if (CONFIG.cometEnabled && cometParticles.length > 0) {
      const rgb = CONFIG.lineColor || [255, 255, 255];
      const damp = Math.max(0, 1 - CONFIG.cometDamping * dt);
      const dms = dt * (1000 / 60);
      const prevComp = ctx.globalCompositeOperation;
      ctx.globalCompositeOperation = 'lighter';
      for (let i = cometParticles.length - 1; i >= 0; i--) {
        const p = cometParticles[i];
        // 物理更新
        p.vx *= damp; p.vy *= damp;
        p.x += p.vx * dt; p.y += p.vy * dt;
        p.life -= dms;
        if (p.life <= 0) { cometParticles.splice(i, 1); continue; }
        // 绘制
        const k = p.life / p.lifeMax; // 0..1
        const alpha = Math.max(0, k * CONFIG.cometOpacity);
        const rr = Math.max(0.5, p.r * (0.4 + 0.6 * k));
        ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha.toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, rr, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalCompositeOperation = prevComp;
    }
    // 下一帧（放在渲染之后，避免并发帧）
    raf = window.requestAnimationFrame(loop);
  }

  // Reduced motion + visibility handling
  const media = window.matchMedia('(prefers-reduced-motion: reduce)');
  function pause() { if (raf) { window.cancelAnimationFrame(raf); raf = 0; } }
  function start() { if (!raf) { lastTime = performance.now(); raf = window.requestAnimationFrame(loop); } }
  function handleVisibility() { if (document.hidden) pause(); else if (isDark && !media.matches) start(); }

  // Mode application
  function applyMode() {
    const prev = isDark; isDark = root.classList.contains('dark');
    if (!isDark || media.matches) {
      canvas.style.display = 'none';
      pause(); ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      return;
    }
    canvas.style.display = 'block';
    resize(); if (!raf) start();
  }

  const mo = new MutationObserver(applyMode);
  mo.observe(root, { attributes: true, attributeFilter: ['class'] });

  window.addEventListener('resize', () => { if (isDark) { resize(); } }, { passive: true });
  document.addEventListener('visibilitychange', handleVisibility);
  media.addEventListener ? media.addEventListener('change', applyMode) : media.addListener(applyMode);

  // Init
  initWebGPU();
  resize(); computeTargetCount(); ensureCount(); applyMode();
})();
