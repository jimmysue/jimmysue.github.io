/* Tiny click-to-zoom lightbox for paper figures.
   Attaches to <figure> images and any <img> inside <main>. Esc / click overlay to close. */
(function () {
  let overlay = null;

  function open(src, alt) {
    if (overlay) close();
    overlay = document.createElement('div');
    overlay.className = 'lightbox-overlay';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.innerHTML =
      '<button class="lightbox-close" aria-label="Close (Esc)">&times;</button>' +
      '<img src="' + src + '" alt="' + (alt || '').replace(/"/g, '&quot;') + '">';
    overlay.addEventListener('click', close);
    document.body.appendChild(overlay);
    document.documentElement.style.overflow = 'hidden';
    requestAnimationFrame(function () { overlay.classList.add('open'); });
  }

  function close() {
    if (!overlay) return;
    const el = overlay;
    overlay = null;
    el.classList.remove('open');
    setTimeout(function () { el.remove(); }, 200);
    document.documentElement.style.overflow = '';
  }

  function attach() {
    const imgs = document.querySelectorAll('figure img, main p img');
    imgs.forEach(function (img) {
      if (img.dataset.lightboxBound) return;
      img.dataset.lightboxBound = '1';
      img.classList.add('lightbox-trigger');
      img.addEventListener('click', function (e) {
        e.stopPropagation();
        open(img.currentSrc || img.src, img.alt);
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attach);
  } else {
    attach();
  }

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') close();
  });
})();
