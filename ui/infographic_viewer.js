// Interactive infographic: shows the server-generated SVG, highlights a hovered item together with
// its connector, shows a small factual tooltip, and reports clicks back to Python.
// Everything displayed here is produced and XML-escaped in Python (src/infographic_renderers.py);
// no secrets and no model output are interpreted as markup.
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const root = parentElement.querySelector('.ig');
  const stage = root.querySelector('.stage');
  const tip = root.querySelector('.tip');
  if (data.max_width) root.style.maxWidth = data.max_width + 'px';
  let mode = null;

  const mark = () => stage.querySelectorAll('[data-id]').forEach((el) =>
    el.classList.toggle('sel', !!data.selected && el.dataset.id === data.selected));

  const render = () => {
    const narrow = !!data.svg_narrow && root.clientWidth < 640;
    const next = narrow ? 'n' : 'w';
    if (next === mode) return;
    mode = next;
    stage.innerHTML = narrow ? data.svg_narrow : data.svg_wide;
    mark();
  };

  const group = (id) => stage.querySelectorAll('[data-id="' + id.replace(/"/g, '\\"') + '"]');
  const clear = () => {
    root.classList.remove('hover');
    stage.querySelectorAll('.hl').forEach((x) => x.classList.remove('hl'));
    tip.style.display = 'none';
  };

  stage.addEventListener('mousemove', (e) => {
    const g = e.target.closest('[data-id]');
    if (!g || g.dataset.id === 'center') { clear(); return; }
    const id = g.dataset.id;
    if (!root.classList.contains('hover') || !g.classList.contains('hl')) {
      stage.querySelectorAll('.hl').forEach((x) => x.classList.remove('hl'));
      group(id).forEach((x) => x.classList.add('hl'));
      root.classList.add('hover');
    }
    const d = (data.details || {})[id];
    if (!d) return;
    tip.replaceChildren();
    const h = document.createElement('div'); h.className = 'tt'; h.textContent = d.title;
    const p = document.createElement('div'); p.textContent = d.text;
    const w = document.createElement('div'); w.className = 'tw'; w.textContent = d.where;
    tip.append(h, p, w);
    const r = root.getBoundingClientRect();
    tip.style.display = 'block';
    tip.style.left = Math.max(0, Math.min(e.clientX - r.left + 14, r.width - 270)) + 'px';
    tip.style.top = Math.max(0, e.clientY - r.top + 16) + 'px';
  });
  stage.addEventListener('mouseleave', clear);
  stage.addEventListener('click', (e) => {
    const g = e.target.closest('[data-id]');
    if (g && g.dataset.id !== 'center') setTriggerValue('selected', g.dataset.id);
  });

  const ro = new ResizeObserver(render);
  ro.observe(root);
  render();
  return () => ro.disconnect();
}
