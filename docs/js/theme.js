/* banknim — theme, Chart.js colors, lightbox */

(function () {
  'use strict';

  function getSystemTheme() {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  function getSavedTheme() {
    try { return localStorage.getItem('banknim-theme'); } catch (e) { return null; }
  }
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem('banknim-theme', theme); } catch (e) {}
    var btn = document.getElementById('themeToggle');
    if (btn) btn.textContent = theme === 'dark' ? '\u2600' : '\u263E';
  }

  applyTheme(getSavedTheme() || 'light');

  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function (e) {
    if (!getSavedTheme()) applyTheme(e.matches ? 'dark' : 'light');
  });

  window.rcToggleTheme = function () {
    var cur = document.documentElement.getAttribute('data-theme') || getSystemTheme();
    applyTheme(cur === 'dark' ? 'light' : 'dark');
    if (window.rcRebuildCharts) window.rcRebuildCharts();
  };

  window.rcIsDark = function () {
    return document.documentElement.getAttribute('data-theme') === 'dark';
  };

  window.rcColors = function () {
    var dk = window.rcIsDark();
    return {
      navy:    dk ? '#c5cfe0' : '#1a1a2e',
      accent:  dk ? '#7ba0e0' : '#537ec5',
      red:     dk ? '#ef5350' : '#c0392b',
      green:   dk ? '#66bb6a' : '#27ae60',
      orange:  dk ? '#ffa726' : '#e67e22',
      purple:  dk ? '#ab47bc' : '#8e44ad',
      teal:    dk ? '#26a69a' : '#16a085',
      slate:   dk ? '#90a4ae' : '#2c3e50',
      deepOr:  dk ? '#ff7043' : '#d35400',
      muted:   dk ? '#3a4050' : '#d1d5db',
      text:    dk ? '#d4d8e0' : '#1f2937',
      textSec: dk ? '#9ca3af' : '#6b7280',
      grid:    dk ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)',
      bg:      dk ? '#1a1e2c' : '#ffffff',
      palette: dk
        ? ['#7ba0e0','#ef5350','#66bb6a','#ffa726','#ab47bc','#90a4ae','#26a69a','#ff7043','#42a5f5','#ce93d8']
        : ['#537ec5','#c0392b','#27ae60','#e67e22','#8e44ad','#2c3e50','#16a085','#d35400','#2980b9','#9b59b6']
    };
  };

  window.rcChartDefaults = function () {
    if (typeof Chart === 'undefined') return;
    try {
      var c = window.rcColors();
      Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
      Chart.defaults.font.size = 13;
      Chart.defaults.color = c.textSec;
      Chart.defaults.borderColor = c.grid;
      if (Chart.defaults.plugins.legend && Chart.defaults.plugins.legend.labels)
        Chart.defaults.plugins.legend.labels.color = c.textSec;
      if (Chart.defaults.plugins.tooltip) {
        Chart.defaults.plugins.tooltip.backgroundColor = c.navy;
        Chart.defaults.plugins.tooltip.titleColor = '#fff';
        Chart.defaults.plugins.tooltip.bodyColor = '#e0e0e0';
      }
    } catch (e) {}
  };

  /* -- Lightbox -- */
  window.rcOpenLightbox = function (src, alt) {
    var ov = document.getElementById('lightboxOverlay');
    var im = document.getElementById('lightboxImg');
    var cap = document.getElementById('lightboxCaption');
    if (!ov || !im) return;
    im.src = src; im.alt = alt || '';
    if (cap) cap.textContent = alt || '';
    ov.classList.add('active');
    document.body.style.overflow = 'hidden';
  };
  window.rcCloseLightbox = function () {
    var ov = document.getElementById('lightboxOverlay');
    if (!ov) return;
    ov.classList.remove('active');
    document.body.style.overflow = '';
  };

  /* -- Scroll-spy -- */
  window.rcInitScrollSpy = function () {
    var secs = document.querySelectorAll('section[id]');
    var links = document.querySelectorAll('.page-nav a');
    if (!secs.length || !links.length) return;
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          links.forEach(function (a) { a.classList.remove('active'); });
          var t = document.querySelector('.page-nav a[href="#' + e.target.id + '"]');
          if (t) t.classList.add('active');
        }
      });
    }, { rootMargin: '-80px 0px -60% 0px', threshold: 0 });
    secs.forEach(function (s) { obs.observe(s); });
  };

  /* -- Mobile nav + keyboard -- */
  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.nav-links a').forEach(function (a) {
      a.addEventListener('click', function () {
        var nl = document.querySelector('.nav-links');
        if (nl) nl.classList.remove('open');
      });
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') rcCloseLightbox();
    });
  });
})();
