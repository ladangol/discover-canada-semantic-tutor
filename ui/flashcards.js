// Flip-card deck: click/tap or Space to flip, ← → to move, "Got it" / "Still learning" to track progress.
// Built entirely with createElement + textContent, so card text can never be interpreted as markup.
export default function (component) {
  const { data, parentElement } = component;
  const root = parentElement.querySelector('.fc');
  root.style.setProperty('--accent', data.accent || '#10A89A');
  const cards = data.cards || [];
  let i = 0, flipped = false;
  const known = new Set(), learning = new Set();

  const SVG = 'http://www.w3.org/2000/svg';
  const PATHS = {
    check: 'M9 16.17 4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z',
    repeat: 'M7 7h10v3l4-4-4-4v3H5v6h2V7zm10 10H7v-3l-4 4 4 4v-3h12v-6h-2v4z',
    trophy: 'M19 5h-2V3H7v2H5c-1.1 0-2 .9-2 2v1c0 2.55 1.92 4.63 4.39 4.94A5.01 5.01 0 0 0 11 15.9V19H7v2h10v-2h-4v-3.1a5.01 5.01 0 0 0 3.61-2.96C19.08 12.63 21 10.55 21 8V7c0-1.1-.9-2-2-2zM5 8V7h2v3.82C5.84 10.4 5 9.3 5 8zm14 0c0 1.3-.84 2.4-2 2.82V7h2v1z',
    thumb: 'M1 21h4V9H1v12zm22-11c0-1.1-.9-2-2-2h-6.31l.95-4.57.03-.32c0-.41-.17-.79-.44-1.06L14.17 1 7.59 7.59C7.22 7.95 7 8.45 7 9v10c0 1.1.9 2 2 2h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73v-2z'
  };
  const icon = (name, size) => {
    const s = document.createElementNS(SVG, 'svg');
    s.setAttribute('viewBox', '0 0 24 24'); s.setAttribute('width', size || 18); s.setAttribute('height', size || 18);
    s.setAttribute('fill', 'currentColor'); s.setAttribute('aria-hidden', 'true'); s.classList.add('ic');
    const p = document.createElementNS(SVG, 'path'); p.setAttribute('d', PATHS[name]); s.append(p);
    return s;
  };
  const withIcon = (name, label) => { const f = document.createDocumentFragment(); f.append(icon(name), document.createTextNode(' ' + label)); return f; };

  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };

  const draw = () => {
    root.replaceChildren();
    if (!cards.length) { root.append(el('div', 'done', 'No cards yet.')); return; }
    const finished = known.size + learning.size === cards.length && i >= cards.length;
    if (i >= cards.length) {
      const box = el('div', 'done');
      const big = el('div', 'big'); big.append(icon(known.size === cards.length ? 'trophy' : 'check', 56));
      box.append(big,
        el('h3', '', known.size === cards.length ? 'Deck complete!' : 'Nice round!'),
        el('p', '', `${known.size} got it · ${learning.size} still learning`));
      const again = el('button', 'btn go', learning.size ? 'Review the tricky ones' : 'Restart deck');
      again.onclick = () => {
        const keep = learning.size ? [...learning] : cards.map((_, k) => k);
        known.clear(); learning.clear();
        order = keep; i = 0; flipped = false; draw();
      };
      box.append(again);
      root.append(box);
      return;
    }
    const c = cards[order[i]];
    // progress
    const top = el('div', 'top');
    const bar = el('div', 'bar'); const fill = el('i'); fill.style.width = (100 * i / order.length) + '%'; bar.append(fill);
    top.append(bar, el('span', 'count', `${i + 1} / ${order.length}`));
    // card
    const scene = el('div', 'scene');
    const card = el('div', 'card3d' + (flipped ? ' flip' : ''));
    const front = el('div', 'face front'); front.append(el('span', 'tag', 'Question'), el('div', 'txt', c.front));
    const back = el('div', 'face back'); back.append(el('span', 'tag', 'Answer'), el('div', 'txt', c.back));
    if (c.meta) back.append(el('div', 'meta', c.meta));
    card.append(front, back);
    scene.append(card);
    scene.onclick = () => { flipped = !flipped; card.classList.toggle('flip', flipped); };
    // controls
    const ctrl = el('div', 'ctrl');
    const mk = (cls, label, fn, iconName) => { const b = el('button', 'btn ' + cls); b.append(iconName ? withIcon(iconName, label) : document.createTextNode(label)); b.onclick = (e) => { e.stopPropagation(); fn(); }; return b; };
    const go = (d) => { i = Math.max(0, i + d); flipped = false; draw(); };
    const mark = (set) => { set.add(order[i]); (set === known ? learning : known).delete(order[i]); i += 1; flipped = false; draw(); };
    ctrl.append(mk('again', 'Still learning', () => mark(learning), 'repeat'), mk('nav', '←', () => go(-1)),
                mk('nav', '→', () => go(1)), mk('got', 'Got it', () => mark(known), 'check'));
    const tally = el('div', 'tally');
    const k = el('span', 'k'); k.append(icon('check', 16), document.createTextNode(` ${known.size} got it`));
    const l = el('span', 'l'); l.append(icon('repeat', 16), document.createTextNode(` ${learning.size} to review`));
    tally.append(k, l,
                 el('span', 'h', 'Tap the card to flip · ← → keys to move'));
    root.append(top, scene, ctrl, tally);
  };

  let order = cards.map((_, k) => k);
  root.onkeydown = (e) => {
    if (e.key === 'ArrowRight' && i < cards.length) { i += 1; flipped = false; draw(); }
    else if (e.key === 'ArrowLeft' && i > 0) { i -= 1; flipped = false; draw(); }
    else if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); flipped = !flipped; const c3 = root.querySelector('.card3d'); if (c3) c3.classList.toggle('flip', flipped); }
    else return;
    root.focus();
  };
  draw();
}
