/*
 * Starfield background animation
 * Lightweight canvas-based stars with parallax, twinkle and sporadic shooting stars.
 */
(function () {
  if (typeof window === 'undefined' || typeof document === 'undefined') return;

  const existing = document.getElementById('starfield');
  if (existing) return; // avoid duplicates

  const canvas = document.createElement('canvas');
  canvas.id = 'starfield';
  document.body.prepend(canvas);

  const ctx = canvas.getContext('2d');
  let width = 0, height = 0, dpr = Math.min(window.devicePixelRatio || 1, 2);

  // Configs
  const STAR_DENSITY = 0.12; // stars per 1000 px^2 (dark mode only)
  const LAYERS = [
    { speed: 0.02, size: [0.4, 0.9], alpha: [0.35, 0.7] },
    { speed: 0.05, size: [0.6, 1.2], alpha: [0.45, 0.85] },
    { speed: 0.09, size: [0.8, 1.6], alpha: [0.55, 1.0] }
  ];
  const TWINKLE_RATE = 0.02; // chance per frame
  const SHOOTING_STAR_RATE = 0.0015; // chance per frame
  const SHOOTING_STAR_SPEED = 10; // px per frame @1x

  const rand = (a, b) => a + Math.random() * (b - a);
  const choice = arr => arr[(Math.random() * arr.length) | 0];

  // Theme mode detection: dark vs light
  const root = document.documentElement;
  let isDark = root.classList.contains('dark');

  function resize() {
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (isDark) spawnStars();
  }

  // Star entity
  let stars = [];
  function spawnStars() {
    const areaK = (width * height) / 1000;
    const density = isDark ? STAR_DENSITY : 0; // no stars in light mode
    const target = Math.max(0, Math.floor(areaK * density));
    stars = [];
    for (let i = 0; i < target; i++) {
      const layer = choice(LAYERS);
      stars.push({
        x: Math.random() * width,
        y: Math.random() * height,
        r: rand(layer.size[0], layer.size[1]),
        a: rand(layer.alpha[0], layer.alpha[1]),
        v: layer.speed * rand(0.75, 1.25),
        twinkle: Math.random(),
        layer
      });
    }
  }

  // Shooting star entity
  let meteors = [];
  function spawnMeteor() {
    // Start near top with random x and diagonal direction
    const fromLeft = Math.random() < 0.5;
    const startX = fromLeft ? rand(-width * 0.2, width * 0.4) : rand(width * 0.6, width * 1.2);
    const startY = rand(-height * 0.1, height * 0.3);
    const angle = fromLeft ? rand(20, 35) : rand(145, 160); // degrees
    const speed = SHOOTING_STAR_SPEED * rand(0.8, 1.4);
    const vx = Math.cos((angle * Math.PI) / 180) * speed;
    const vy = Math.sin((angle * Math.PI) / 180) * speed;
    meteors.push({ x: startX, y: startY, vx, vy, life: rand(40, 75), trail: [] });
  }

  function drawBackground() {
    // Subtle vertical gradient for depth
    if (isDark) {
      const g = ctx.createLinearGradient(0, 0, 0, height);
      g.addColorStop(0, 'rgba(0, 10, 20, 1)');
      g.addColorStop(1, 'rgba(0, 0, 0, 1)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, width, height);
    } else {
      // In light mode, we don't paint anything; keep fully transparent
      ctx.clearRect(0, 0, width, height);
    }
  }

  function drawStars() {
    for (let s of stars) {
      s.x += s.v; // drift horizontally
      if (s.x > width + 2) s.x = -2;

      // twinkle
      if (Math.random() < TWINKLE_RATE) {
        s.twinkle += rand(-0.15, 0.15);
        if (s.twinkle < 0) s.twinkle = 0;
        if (s.twinkle > 1) s.twinkle = 1;
      }
      const alpha = s.a * (0.6 + 0.4 * s.twinkle);

      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  function drawMeteors() {
    const next = [];
    for (let m of meteors) {
      m.x += m.vx;
      m.y += m.vy;
      m.life -= 1;
      // trail
      m.trail.push({ x: m.x, y: m.y });
      if (m.trail.length > 12) m.trail.shift();

      // draw trail
      for (let i = m.trail.length - 1; i > 0; i--) {
        const p = m.trail[i];
        const p2 = m.trail[i - 1];
        const t = i / m.trail.length;
        ctx.strokeStyle = `rgba(255,255,255,${0.15 * t})`;
        ctx.lineWidth = 1.5 * (1 - t);
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }

      // bright head
      ctx.beginPath();
      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      ctx.arc(m.x, m.y, 1.4, 0, Math.PI * 2);
      ctx.fill();

      if (m.life > 0 && m.x > -200 && m.x < width + 200 && m.y < height + 200) {
        next.push(m);
      }
    }
    meteors = next;
  }

  let rafId = 0;
  function frame() {
    drawBackground();
    if (isDark) {
      drawStars();
      drawMeteors();
      if (Math.random() < SHOOTING_STAR_RATE) spawnMeteor();
    }
    rafId = window.requestAnimationFrame(frame);
  }

  // Respect reduced motion
  const media = window.matchMedia('(prefers-reduced-motion: reduce)');
  function start() {
    if (media.matches) return; // do nothing if user prefers reduced motion
    cancel();
    rafId = window.requestAnimationFrame(frame);
  }
  function cancel() {
    if (rafId) window.cancelAnimationFrame(rafId);
    rafId = 0;
  }

  // Observe theme mode changes by watching html.class changes
  const mo = new MutationObserver(() => {
    applyMode();
  });
  mo.observe(root, { attributes: true, attributeFilter: ['class'] });

  function applyMode() {
    const wasDark = isDark;
    isDark = root.classList.contains('dark');
    if (!isDark || media.matches) {
      // Light mode or reduced motion: hide canvas and stop animating
      canvas.style.display = 'none';
      cancel();
      ctx.clearRect(0, 0, width, height);
      return;
    }
    // Dark mode: show and ensure sized + stars
    canvas.style.display = 'block';
    resize();
    if (!rafId) start();
  }

  window.addEventListener('resize', () => { if (isDark) resize(); }, { passive: true });
  const onReduceChange = () => applyMode();
  media.addEventListener ? media.addEventListener('change', onReduceChange) : media.addListener(onReduceChange);

  // Initialize mode and sizing
  resize();
  applyMode();
})();
