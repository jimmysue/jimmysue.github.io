// Full-text search modal backed by Pagefind. Initialised lazily on first open
// so the Pagefind bundle isn't fetched on pages where search is never used.
(function () {
  var modal = document.getElementById("search-modal");
  var openBtn = document.getElementById("search-open");
  if (!modal || !openBtn) return;

  var inited = false;
  function init() {
    if (inited) return;
    if (typeof PagefindUI === "undefined") return; // bundle missing → degrade silently
    // Resolve the depth-relative path (e.g. "pagefind/" or "../../pagefind/") to an
    // absolute path. PagefindUI feeds bundlePath into a dynamic import(); a bare
    // specifier like "pagefind/pagefind.js" fails ES module resolution, so we must
    // hand it an absolute ("/pagefind/") or "./"-prefixed path.
    var rawPath = modal.getAttribute("data-pagefind-path") || "pagefind/";
    var bundlePath = new URL(rawPath, window.location.href).pathname;
    new PagefindUI({ element: "#search", bundlePath: bundlePath, showImages: false, resetStyles: false });
    inited = true;
  }

  function open() {
    init();
    modal.hidden = false;
    document.body.classList.add("search-open");
    openBtn.setAttribute("aria-expanded", "true");
    var input = modal.querySelector("input");
    if (input) input.focus();
  }
  function close() {
    modal.hidden = true;
    document.body.classList.remove("search-open");
    openBtn.setAttribute("aria-expanded", "false");
  }

  openBtn.addEventListener("click", open);
  modal.querySelectorAll("[data-search-close]").forEach(function (el) {
    el.addEventListener("click", close);
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !modal.hidden) close();
  });
})();
