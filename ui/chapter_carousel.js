// 3D chapter carousel: the active card sits in front, neighbours tilt and shrink behind it.
// Tap a side card (or use the arrows, ← →, or swipe) to bring it forward; tap the front card, press Enter,
// or press "Open chapter" to send the chapter name back to Python. Built with createElement + textContent only.
const MAX_VISIBLE = 3;

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const root = parentElement.querySelector('.cc');
  const items = data.items || [];
  if (!items.length) return;

  let active = Number.isInteger(parentElement.__active) ? parentElement.__active : 0;   // survives reruns
  active = Math.min(Math.max(active, 0), items.length - 1);

  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const icon = (name) => { const s = el('span', 'mi', name); s.setAttribute('aria-hidden', 'true'); return s; };

  root.replaceChildren();
  const stage = el('div', 'stage');
  const cards = items.map((it, i) => {
    const card = el('div', 'card');
    card.style.setProperty('--a', it.a); card.style.setProperty('--b', it.b);
    const chip = el('div', 'chip'); chip.append(icon(it.icon));
    card.append(chip, el('div', 'num', String(i + 1).padStart(2, '0')), el('h3', '', it.title), el('div', 'go', 'Open chapter  →'));
    card.addEventListener('click', () => (i === active ? open() : go(i)));
    stage.append(card);
    return card;
  });

  const prev = el('button', 'arrow prev'); prev.append(icon('chevron_left')); prev.setAttribute('aria-label', 'Previous chapter');
  const next = el('button', 'arrow next'); next.append(icon('chevron_right')); next.setAttribute('aria-label', 'Next chapter');
  prev.onclick = () => go(active - 1); next.onclick = () => go(active + 1);

  const dots = el('div', 'dots');
  const dotEls = items.map((it, i) => { const d = el('button', 'dot'); d.setAttribute('aria-label', it.title); d.onclick = () => go(i); dots.append(d); return d; });

  const cta = el('button', 'cta');
  cta.onclick = () => open();

  root.append(stage, prev, next, dots, cta);
  root.tabIndex = 0;

  function open() { setTriggerValue('open', items[active].title); }

  function go(i) {
    active = Math.min(Math.max(i, 0), items.length - 1);
    parentElement.__active = active;
    paint();
  }

  function paint() {
    cards.forEach((card, i) => {
      const d = i - active, ad = Math.abs(d);
      const shown = ad <= MAX_VISIBLE;
      card.classList.toggle('on', d === 0);
      card.style.setProperty('--d', d);
      card.style.setProperty('--ad', ad);
      card.style.setProperty('--sign', Math.sign(d));
      card.style.opacity = shown ? (ad === 0 ? 1 : Math.max(0.25, 1 - ad * 0.28)) : 0;
      card.style.pointerEvents = shown ? 'auto' : 'none';
      card.style.zIndex = String(100 - ad);
      card.setAttribute('aria-hidden', shown ? 'false' : 'true');
    });
    dotEls.forEach((d, i) => d.classList.toggle('on', i === active));
    prev.disabled = active === 0; next.disabled = active === items.length - 1;
    cta.replaceChildren(icon('arrow_forward'), document.createTextNode(' Open "' + items[active].title + '"'));
  }

  root.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') { go(active + 1); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { go(active - 1); e.preventDefault(); }
    else if (e.key === 'Enter') { open(); e.preventDefault(); }
  });

  // swipe (touch or mouse drag); a drag shorter than 8px is treated as a tap and falls through to the click handler
  let x0 = null, moved = false;
  stage.addEventListener('pointerdown', (e) => { x0 = e.clientX; moved = false; });
  stage.addEventListener('pointermove', (e) => { if (x0 !== null && Math.abs(e.clientX - x0) > 8) moved = true; });
  stage.addEventListener('pointerup', (e) => {
    if (x0 === null) return;
    const dx = e.clientX - x0; x0 = null;
    if (Math.abs(dx) > 40) { go(active + (dx < 0 ? 1 : -1)); }
  });
  stage.addEventListener('click', (e) => { if (moved) { e.stopPropagation(); moved = false; } }, true);

  paint();
}
