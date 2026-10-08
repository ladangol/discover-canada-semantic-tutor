// three.js 3-D atlas. Receives {points, selected, neighbours, retrieved, query, emphasize}
// from Python, reports clicks back with setTriggerValue('selected', chunk_id).
// Positions are a 3-D UMAP projection: for looking around only, never for similarity.
const THREE_URL = 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const root = parentElement.querySelector('.a3d');
  const tip = parentElement.querySelector('.tip');
  let disposed = false;
  let cleanup = () => {};

  (async () => {
    let THREE;
    try {
      THREE = await import(THREE_URL);
    } catch (e) {
      root.insertAdjacentText('afterbegin', 'The 3-D view loads three.js from a CDN and needs internet access.');
      return;
    }
    if (disposed) return;

    const view = (window.__atlas3dView = window.__atlas3dView || { rx: 0.35, ry: 0.6, z: 4.6, auto: true });
    const pts = data.points;
    const byId = new Map(pts.map((p, i) => [p.id, i]));

    // normalise coordinates into roughly [-1, 1]^3
    const all = pts.map((p) => [p.x, p.y, p.z]);
    if (data.query) all.push(data.query);
    const mins = [0, 1, 2].map((k) => Math.min(...all.map((a) => a[k])));
    const maxs = [0, 1, 2].map((k) => Math.max(...all.map((a) => a[k])));
    const ext = Math.max(...[0, 1, 2].map((k) => maxs[k] - mins[k])) / 2 || 1;
    const norm = (a) => new THREE.Vector3(...[0, 1, 2].map((k) => (a[k] - (mins[k] + maxs[k]) / 2) / ext));

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    root.prepend(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
    scene.add(new THREE.AmbientLight(0xffffff, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(3, 4, 5);
    scene.add(sun);
    const group = new THREE.Group();
    scene.add(group);

    const retrieved = new Set(data.retrieved || []);
    const neighbours = new Set(data.neighbours || []);
    const emphasize = !!data.emphasize && retrieved.size > 0;

    // all chunks as one instanced mesh (cheap, raycastable)
    const geo = new THREE.SphereGeometry(0.03, 14, 14);
    const mesh = new THREE.InstancedMesh(geo, new THREE.MeshStandardMaterial({ roughness: 0.45 }), pts.length);
    const m4 = new THREE.Matrix4();
    const white = new THREE.Color('#ffffff');
    pts.forEach((p, i) => {
      const color = new THREE.Color(p.v ? p.c : '#cfcfcf');
      if (!p.v) color.lerp(white, 0.3);
      if (emphasize && !retrieved.has(p.id)) color.lerp(white, 0.72);
      const s = !p.v ? 0.6 : emphasize && !retrieved.has(p.id) ? 0.8 : 1;
      m4.compose(norm([p.x, p.y, p.z]), new THREE.Quaternion(), new THREE.Vector3(s, s, s));
      mesh.setMatrixAt(i, m4);
      mesh.setColorAt(i, color);
    });
    group.add(mesh);

    const label = (text, color) => {
      const c = document.createElement('canvas');
      c.width = c.height = 64;
      const g = c.getContext('2d');
      g.fillStyle = color;
      g.beginPath(); g.arc(32, 32, 28, 0, 6.3); g.fill();
      g.fillStyle = '#fff'; g.font = 'bold 34px sans-serif';
      g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText(text, 32, 34);
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
      sp.scale.set(0.13, 0.13, 1);
      sp.renderOrder = 10;
      return sp;
    };
    const at = (id) => norm([pts[byId.get(id)].x, pts[byId.get(id)].y, pts[byId.get(id)].z]);

    (data.retrieved || []).forEach((id, n) => {
      if (!byId.has(id)) return;
      const sp = label(String(n + 1), '#c8102e');
      sp.position.copy(at(id));
      group.add(sp);
    });
    (data.neighbours || []).forEach((id, n) => {
      if (!byId.has(id)) return;
      const ring = new THREE.Mesh(new THREE.SphereGeometry(0.055, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0x111111, wireframe: true }));
      ring.position.copy(at(id));
      group.add(ring);
      const sp = label(String(n + 1), '#111111');
      sp.position.copy(at(id)).add(new THREE.Vector3(0, 0.11, 0));
      group.add(sp);
    });
    if (data.selected && byId.has(data.selected)) {
      const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.075),
        new THREE.MeshStandardMaterial({ color: 0xc8102e, emissive: 0x550000 }));
      m.position.copy(at(data.selected));
      group.add(m);
    }
    if (data.query) {
      const q = norm(data.query);
      const star = new THREE.Mesh(new THREE.OctahedronGeometry(0.1),
        new THREE.MeshStandardMaterial({ color: 0xffc400, emissive: 0x553d00 }));
      star.position.copy(q);
      group.add(star);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(star.geometry),
        new THREE.LineBasicMaterial({ color: 0x111111 }));
      edges.position.copy(q);
      group.add(edges);
      const verts = [];
      (data.retrieved || []).forEach((id) => { if (byId.has(id)) verts.push(q, at(id)); });
      group.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(verts),
        new THREE.LineBasicMaterial({ color: 0xc8102e, transparent: true, opacity: 0.45 })));
    }

    // --- interaction: drag = rotate, wheel = zoom, click = inspect ---------------
    const ray = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const pick = (ev) => {
      const r = renderer.domElement.getBoundingClientRect();
      mouse.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(mouse, camera);
      const hits = ray.intersectObject(mesh).filter((h) => pts[h.instanceId].v);
      return hits.length ? hits[0].instanceId : -1;
    };
    let down = null;
    const el = renderer.domElement;
    el.style.touchAction = 'none';
    el.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY, moved: 0 }; el.setPointerCapture(e.pointerId); });
    el.addEventListener('pointermove', (e) => {
      if (down) {
        const dx = e.clientX - down.x, dy = e.clientY - down.y;
        down.moved += Math.abs(dx) + Math.abs(dy);
        down.x = e.clientX; down.y = e.clientY;
        view.ry += dx * 0.006;
        view.rx = Math.max(-1.5, Math.min(1.5, view.rx + dy * 0.006));
        view.auto = false;
        tip.style.display = 'none';
        return;
      }
      const i = pick(e);
      if (i < 0) { tip.style.display = 'none'; return; }
      const p = pts[i];
      const r = root.getBoundingClientRect();
      tip.textContent = `${p.t}`;
      tip.style.display = 'block';
      tip.style.left = Math.min(e.clientX - r.left + 14, r.width - 270) + 'px';
      tip.style.top = Math.min(e.clientY - r.top + 14, r.height - 110) + 'px';
    });
    el.addEventListener('pointerup', (e) => {
      const wasClick = down && down.moved < 5;
      down = null;
      if (wasClick) {
        const i = pick(e);
        if (i >= 0) setTriggerValue('selected', pts[i].id);
      }
    });
    el.addEventListener('pointerleave', () => { tip.style.display = 'none'; });
    el.addEventListener('wheel', (e) => {
      e.preventDefault();
      view.z = Math.max(1.8, Math.min(9, view.z * (e.deltaY > 0 ? 1.08 : 0.92)));
    }, { passive: false });

    const resize = () => {
      const w = root.clientWidth, h = root.clientHeight;
      renderer.setSize(w, h, true);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(root);
    resize();

    let raf = 0;
    const loop = () => {
      if (disposed) return;
      if (view.auto) view.ry += 0.0018;
      group.rotation.set(view.rx, view.ry, 0);
      camera.position.set(0, 0, view.z);
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    loop();

    cleanup = () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      scene.traverse((o) => { o.geometry && o.geometry.dispose(); });
      renderer.dispose();
      renderer.domElement.remove();
    };
    if (disposed) cleanup();
  })();

  return () => { disposed = true; cleanup(); };
}
