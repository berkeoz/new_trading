// Site menu shared by every page: a slim bar with a sidebar drawer (all pages and the main page's sections).
(function () {
  const MENU = [
    ['Overview', [['Today\'s signals', '/#signals'], ['Signal timeline', '/#timeline'], ['Market breakdown', '/#breakdown'], ['Buy & sell checklist', '/#checklist']]],
    ['Analysis', [['Chart formations', '/patterns'], ['Market brief', '/brief'], ['Twitter analysis', '/x']]],
    ['Personal', [['Finances', '/finance']]],
  ];
  const SECTIONS = {signals: 'p-today', timeline: 'p-timeline', breakdown: 'p-breakdown', checklist: 'p-checklist'};
  const path = location.pathname.replace(/\.html$/, '').replace(/\/$/, '') || '/';
  const onMain = path == '/' || path == '/index';
  const css = `
#nt-bar{position:sticky;top:0;z-index:900;display:flex;align-items:center;gap:10px;padding:6px 14px;background:#11141bee;
  backdrop-filter:blur(6px);border-bottom:1px solid #2a2f3a;font:14px system-ui,Segoe UI,sans-serif;color:#e6e6e6}
#nt-bar button{background:#1c2130;color:#e6e6e6;border:1px solid #2f3546;border-radius:6px;padding:4px 10px;font:inherit;cursor:pointer}
#nt-bar .nt-title{font-weight:600} #nt-bar .nt-sub{color:#8a93a5}
#nt-bar select{background:#1c2130;color:#e6e6e6;border:1px solid #2f3546;border-radius:6px;padding:4px 8px;font:inherit}
#nt-shade{position:fixed;inset:0;background:#0008;z-index:950;display:none}
#nt-drawer{position:fixed;top:0;left:0;bottom:0;width:250px;max-width:85vw;background:#151922;border-right:1px solid #2a2f3a;z-index:960;
  transform:translateX(-100%);transition:transform .18s ease;overflow-y:auto;padding:12px 0;font:14px system-ui,Segoe UI,sans-serif}
#nt-drawer.open{transform:none} #nt-shade.open{display:block}
#nt-drawer .g{color:#8a93a5;font-size:11px;text-transform:uppercase;letter-spacing:.06em;padding:12px 18px 4px}
#nt-drawer a{display:block;padding:7px 18px;color:#e6e6e6;text-decoration:none;border-left:3px solid transparent}
#nt-drawer a:hover{background:#1f2533} #nt-drawer a.cur{border-left-color:#5c8dff;background:#1b2130;color:#9ab6ff}
body.nt-main .tabs{display:none}`;
  const st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);
  const cur = (href) => {
    const [p, h] = href.split('#');
    if (onMain) return (p == '/' || p == '') && h == (location.hash.slice(1) || 'signals');
    return p == path;
  };
  const drawer = document.createElement('nav'); drawer.id = 'nt-drawer';
  const shade = document.createElement('div'); shade.id = 'nt-shade';
  const draw = () => drawer.innerHTML = '<div style="padding:4px 18px 8px;font-weight:600">New Trading</div>' + MENU.map(([g, ls]) =>
    `<div class="g">${g}</div>` + ls.map(([t, h]) => `<a href="${h}" class="${cur(h) ? 'cur' : ''}">${t}</a>`).join('')).join('');
  draw();
  const open = on => { drawer.classList.toggle('open', on); shade.classList.toggle('open', on); };
  shade.onclick = () => open(false);
  drawer.addEventListener('click', e => { if (e.target.tagName == 'A') open(false); });
  const title = (MENU.flatMap(m => m[1]).find(([, h]) => !onMain && h == path) || [document.title])[0];
  const bar = document.createElement('div'); bar.id = 'nt-bar';
  bar.innerHTML = `<button aria-label="Menu" title="Menu">☰ Menu</button><span class="nt-title">${onMain ? 'Dashboard' : title}</span>` +
    (onMain ? `<select aria-label="Section">${MENU[0][1].map(([t, h]) => `<option value="${h.split('#')[1]}">${t}</option>`).join('')}</select>` : '');
  bar.querySelector('button').onclick = () => open(!drawer.classList.contains('open'));
  document.body.prepend(shade); document.body.prepend(drawer); document.body.prepend(bar);
  document.addEventListener('keydown', e => { if (e.key == 'Escape') open(false); });
  if (!onMain) return;
  // main page: sections are the old tabs (kept in the DOM, hidden, so the page's own code still works)
  document.body.classList.add('nt-main');
  const sel = bar.querySelector('select');
  const show = () => {
    const key = location.hash.slice(1) || 'signals', id = SECTIONS[key];
    if (!id || !document.getElementById(id)) return;
    const tab = document.querySelector(`.tab[data-panel="${id}"]`);
    if (tab) tab.click();
    else {
      document.querySelectorAll('.tab,.tab-panel').forEach(e => e.classList.remove('active'));
      document.getElementById(id).classList.add('active');
    }
    sel.value = key; draw();
  };
  sel.onchange = () => { location.hash = sel.value; };
  window.addEventListener('hashchange', show);
  // keep the menu in sync when the page itself switches tabs (e.g. clicking a ticker opens its timeline)
  document.querySelectorAll('.tab[data-panel]').forEach(t => t.addEventListener('click', () => {
    const k = Object.keys(SECTIONS).find(k => SECTIONS[k] == t.dataset.panel);
    if (k && sel.value != k) { sel.value = k; history.replaceState(null, '', '#' + k); draw(); }
  }));
  show();
})();
