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
    maxSpeed: 0.35,
    minSpeed: 0.2,        // 速度下限，防止完全停滞
    damping: 0.01,         // 速度阻尼（每帧按 dt 衰减）
    linkDistance: 120,
    linkWidth: 0.8,
    linkOpacity: 0.7,
    drawMouseLinks: true,
    hoverRadius: 160,
    attractStrength: 0.005, // 鼠标吸引强度（加速度）
    // 近距离互斥（相互排斥），只在非常接近时生效
    repelDistance: 22,       // px
    repelStrength: 0.006,    // 排斥强度
    dotColor: 'rgba(255,255,255,1)',
    lineColor: [255, 255, 255],
    twinkle: true,
    twinkleAmplitude: 0.35
  };

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
  window.addEventListener('mouseenter', () => (mouse.active = true));
  window.addEventListener('mouseleave', () => (mouse.active = false));
  window.addEventListener('touchstart', (e) => { const t = e.touches[0]; if (t) setMouse(t); }, { passive: true });
  window.addEventListener('touchmove', (e) => { const t = e.touches[0]; if (t) setMouse(t); }, { passive: true });
  window.addEventListener('touchend', () => (mouse.active = false));

  // Grid for neighborhood search
  let cellSize = CONFIG.linkDistance;
  function buildGrid() {
    cellSize = CONFIG.linkDistance;
    const cols = Math.ceil(window.innerWidth / cellSize);
    const rows = Math.ceil(window.innerHeight / cellSize);
    const grid = new Array(cols * rows);
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      const cx = Math.max(0, Math.min(cols - 1, (p.x / cellSize) | 0));
      const cy = Math.max(0, Math.min(rows - 1, (p.y / cellSize) | 0));
      const idx = cy * cols + cx;
      (grid[idx] || (grid[idx] = [])).push(i);
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
      const OFFS = [ [0,0], [1,0], [0,1], [1,1], [1,-1] ];
      for (let cy = 0; cy < rows; cy++) {
        for (let cx = 0; cx < cols; cx++) {
          const bucket = grid[cy * cols + cx];
          if (!bucket) continue;
          for (let k = 0; k < OFFS.length; k++) {
            const nx = cx + OFFS[k][0]; const ny = cy + OFFS[k][1];
            if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) continue;
            const nb = grid[ny * cols + nx]; if (!nb) continue;
            if (OFFS[k][0] === 0 && OFFS[k][1] === 0) {
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

    // Linking using grid
    const { grid, cols, rows } = buildGrid();
    const maxD = CONFIG.linkDistance; const maxD2 = maxD * maxD;
    const [lr, lg, lb] = CONFIG.lineColor; ctx.lineWidth = CONFIG.linkWidth;

    function drawLink(a, b) {
      const dx = a.x - b.x; const dy = a.y - b.y; const d2 = dx * dx + dy * dy; if (d2 > maxD2) return;
      const d = Math.sqrt(d2); const alpha = (1 - d / maxD) * CONFIG.linkOpacity;
      ctx.strokeStyle = `rgba(${lr},${lg},${lb},${alpha.toFixed(3)})`;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }

    for (let cy = 0; cy < rows; cy++) {
      for (let cx = 0; cx < cols; cx++) {
        const cellIdx = cy * cols + cx; const bucket = grid[cellIdx]; if (!bucket) continue;
        // within-cell pairs
        for (let ii = 0; ii < bucket.length; ii++) {
          for (let jj = ii + 1; jj < bucket.length; jj++) drawLink(particles[bucket[ii]], particles[bucket[jj]]);
        }
        // neighbor cells
        for (let oy = -1; oy <= 1; oy++) {
          for (let ox = -1; ox <= 1; ox++) {
            if (ox === 0 && oy === 0) continue; const nx = cx + ox, ny = cy + oy;
            if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) continue; const nb = grid[ny * cols + nx]; if (!nb) continue;
            for (let ii = 0; ii < bucket.length; ii++) { for (let jj = 0; jj < nb.length; jj++) drawLink(particles[bucket[ii]], particles[nb[jj]]); }
          }
        }
      }
    }

    // Mouse links
    if (mouse.active && CONFIG.drawMouseLinks) {
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i]; const dx = p.x - mouse.x; const dy = p.y - mouse.y; const d2 = dx * dx + dy * dy;
        if (d2 <= maxD2) {
          const d = Math.sqrt(d2); const alpha = (1 - d / maxD) * (CONFIG.linkOpacity * 0.9);
          ctx.strokeStyle = `rgba(${lr},${lg},${lb},${alpha.toFixed(3)})`;
          ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(mouse.x, mouse.y); ctx.stroke();
        }
      }
    }

    // Draw particles
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i]; let alpha = 1;
      if (CONFIG.twinkle) { p.t += 0.02; alpha = 1 - (Math.sin(p.t) * 0.5 + 0.5) * CONFIG.twinkleAmplitude; }
      ctx.fillStyle = CONFIG.dotColor.replace(/0\.9\)/, `${(0.75 + 0.25 * alpha).toFixed(3)})`);
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
