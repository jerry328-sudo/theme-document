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
    twinkleAmplitude: 0.35
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
  const mouse = { x: 0, y: 0, active: false };
  function setMouse(e) {
    const rect = canvas.getBoundingClientRect();
    mouse.x = e.clientX - rect.left;
    mouse.y = e.clientY - rect.top;
    mouse.active = true;
  }
  window.addEventListener('mousemove', setMouse, { passive: true });
  window.addEventListener('mouseenter', () => (mouse.active = true), { passive: true });
  window.addEventListener('mouseleave', () => (mouse.active = false), { passive: true });
  // 更可靠的离开检测：当 relatedTarget 为空时，表示鼠标离开了页面
  window.addEventListener('mouseout', (e) => { if (!e.relatedTarget && !e.toElement) mouse.active = false; }, { passive: true });
  document.addEventListener('mouseleave', () => (mouse.active = false), { passive: true });
  window.addEventListener('blur', () => (mouse.active = false));
  window.addEventListener('touchstart', (e) => { const t = e.touches[0]; if (t) setMouse(t); }, { passive: true });
  window.addEventListener('touchmove', (e) => { const t = e.touches[0]; if (t) setMouse(t); }, { passive: true });
  window.addEventListener('touchend', () => (mouse.active = false));
  window.addEventListener('touchcancel', () => (mouse.active = false));

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
