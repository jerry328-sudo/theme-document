/*
 * Constellation Particles Background
 * - Particles link when near, with mouse attraction
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
    maxSpeed: 1,
    minSpeed: 0.3,        // 速度下限，防止完全停滞
    damping: 0.01,         // 速度阻尼（每帧按 dt 衰减）
    linkDistance: 120,
    linkWidth: 0.8,
    linkOpacity: 0.7,
    drawMouseLinks: true,
    hoverRadius: 160,
    attractStrength: 0.005, // 鼠标吸引强度（加速度）
    // 近距离互斥（相互排斥），只在非常接近时生效
    repelDistance: 40,       // px
    repelStrength: 0.006,    // 排斥强度
    dotColor: 'rgba(255,255,255,1)',
    lineColor: [255, 255, 255],
    twinkle: true,
    twinkleAmplitude: 0.35,
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
  const rand = (a, b) => Math.random() * (b - a) + a;

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
  }

  function spawnParticle() {
    const r = rand(CONFIG.dotRadius[0], CONFIG.dotRadius[1]);
    particles.push({
      x: rand(0, window.innerWidth),
      y: rand(0, window.innerHeight),
      vx: rand(-CONFIG.maxSpeed, CONFIG.maxSpeed),
      vy: rand(-CONFIG.maxSpeed, CONFIG.maxSpeed),
      r,
      t: Math.random() * Math.PI * 2
    });
  }

  function ensureCount() {
    while (particles.length < targetCount) spawnParticle();
    if (particles.length > targetCount) particles.length = targetCount;
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

  // Animation loop
  let raf = 0;
  let lastTime = performance.now();
  function loop(now) {
    raf = window.requestAnimationFrame(loop);
    const dt = Math.max(0.5, Math.min(2.5, (now - lastTime) / (1000 / 60)));
    lastTime = now;

    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);

    // 近距离排斥：对非常接近的粒子施加对称的分离冲量
    (function applyRepulsion() {
      const R = CONFIG.repelDistance; const R2 = R * R;
      const { grid, cols, rows } = buildGrid();
      // 半平面邻域以避免重复；(0,0)格用 i<j
      for (let cy = 0; cy < rows; cy++) {
        for (let cx = 0; cx < cols; cx++) {
          const bucket = grid[cy * cols + cx];
          if (!bucket) continue;
          for (let k = 0; k < NEIGHBOR_OFFS.length; k++) {
            const nx = cx + NEIGHBOR_OFFS[k][0]; const ny = cy + NEIGHBOR_OFFS[k][1];
            if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) continue;
            const nb = grid[ny * cols + nx]; if (!nb) continue;
            if (NEIGHBOR_OFFS[k][0] === 0 && NEIGHBOR_OFFS[k][1] === 0) {
              for (let i = 0; i < bucket.length; i++) {
                for (let j = i + 1; j < bucket.length; j++) repelPair(particles[bucket[i]], particles[bucket[j]]);
              }
            } else {
              for (let i = 0; i < bucket.length; i++) {
                for (let j = 0; j < nb.length; j++) repelPair(particles[bucket[i]], particles[nb[j]]);
              }
            }
          }
        }
      }
      function repelPair(a, b) {
        const dx = b.x - a.x; const dy = b.y - a.y; const d2 = dx*dx + dy*dy;
        if (d2 <= 1e-6 || d2 > R2) return;
        const d = Math.sqrt(d2);
        // 线性随距离减弱，越近排斥越强
        const s = (1 - d / R) * CONFIG.repelStrength * dt;
        const ux = dx / d, uy = dy / d;
        // 对称相反方向
        a.vx -= ux * s; a.vy -= uy * s;
        b.vx += ux * s; b.vy += uy * s;
      }
    })();

    // Physics + attraction
    const hr2 = CONFIG.hoverRadius * CONFIG.hoverRadius;
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];

      // 鼠标吸引（加速度）
      if (mouse.active) {
        const dx = mouse.x - p.x; const dy = mouse.y - p.y; const d2 = dx * dx + dy * dy;
        if (d2 < hr2 && d2 > 0.001) {
          const a = CONFIG.attractStrength;
          p.vx += dx * a * dt; p.vy += dy * a * dt;
        }
      }

      // 速度阻尼（指数衰减）
      const damp = Math.max(0, 1 - CONFIG.damping * dt);
      p.vx *= damp; p.vy *= damp;

      // 速度阈值控制：限制上限，抬升下限
      let sp = Math.hypot(p.vx, p.vy);
      if (sp > CONFIG.maxSpeed) {
        const k = CONFIG.maxSpeed / sp; p.vx *= k; p.vy *= k; sp = CONFIG.maxSpeed;
      } else if (sp < CONFIG.minSpeed) {
        // 若几乎静止，则赋予一个随机的微弱方向速度
        if (sp < 1e-4) {
          const ang = Math.random() * Math.PI * 2;
          p.vx = Math.cos(ang) * CONFIG.minSpeed; p.vy = Math.sin(ang) * CONFIG.minSpeed;
        } else {
          const k = CONFIG.minSpeed / sp; p.vx *= k; p.vy *= k;
        }
      }

      // 位置更新
      p.x += p.vx * dt; p.y += p.vy * dt;
      if (p.x < 0) { p.x = 0; p.vx *= -1; } else if (p.x > window.innerWidth) { p.x = window.innerWidth; p.vx *= -1; }
      if (p.y < 0) { p.y = 0; p.vy *= -1; } else if (p.y > window.innerHeight) { p.y = window.innerHeight; p.vy *= -1; }
    }

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
  resize(); computeTargetCount(); ensureCount(); applyMode();
})();
