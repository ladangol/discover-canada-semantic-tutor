// Flip-card deck: click/tap or Space to flip, ← → to move, "Got it" / "Still learning" to track progress.
// Built entirely with createElement + textContent, so card text can never be interpreted as markup.
export default function (component) {
  const { data, parentElement } = component;
  const root = parentElement.querySelector('.fc');
  root.style.setProperty('--accent', data.accent || '#10A89A');
  const cards = data.cards || [];
  let i = 0, flipped = false;
  const known = new Set(), learning = new Set();

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
      box.append(el('div', 'big', known.size === cards.length ? '🎉' : '💪'),
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
    const mk = (cls, label, fn) => { const b = el('button', 'btn ' + cls, label); b.onclick = (e) => { e.stopPropagation(); fn(); }; return b; };
    const go = (d) => { i = Math.max(0, i + d); flipped = false; draw(); };
    const mark = (set) => { set.add(order[i]); (set === known ? learning : known).delete(order[i]); i += 1; flipped = false; draw(); };
    ctrl.append(mk('again', '🔁 Still learning', () => mark(learning)), mk('nav', '←', () => go(-1)),
                mk('nav', '→', () => go(1)), mk('got', '👍 Got it', () => mark(known)));
    const tally = el('div', 'tally');
    tally.append(el('span', 'k', `👍 ${known.size}`), el('span', 'l', `🔁 ${learning.size}`),
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
