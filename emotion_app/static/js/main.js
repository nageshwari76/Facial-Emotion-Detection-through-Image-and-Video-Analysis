/* EmoCare AI — shared UI behaviours */
(function () {
    'use strict';

    // Mobile nav toggle
    var navToggle = document.getElementById('navToggle');
    var navLinks = document.getElementById('navLinks');
    if (navToggle && navLinks) {
        navToggle.addEventListener('click', function () {
            navLinks.classList.toggle('open');
        });
    }

    // Auto-dismiss flash messages
    document.querySelectorAll('.flash').forEach(function (flash) {
        setTimeout(function () {
            flash.style.transition = 'opacity 0.5s';
            flash.style.opacity = '0';
            setTimeout(function () { flash.remove(); }, 500);
        }, 5000);
    });

    // Admin tabs
    var tabButtons = document.querySelectorAll('.tab-btn[data-tab]');
    tabButtons.forEach(function (btn) {
        btn.addEventListener('click', function () {
            var name = btn.getAttribute('data-tab');
            tabButtons.forEach(function (b) { b.classList.remove('active'); });
            btn.classList.add('active');
            document.querySelectorAll('.tab-panel').forEach(function (p) {
                p.classList.toggle('active', p.id === 'tab-' + name);
            });
        });
    });

    // Quick-action buttons that jump to a tab
    document.querySelectorAll('[data-goto]').forEach(function (btn) {
        btn.addEventListener('click', function () {
            var name = btn.getAttribute('data-goto');
            document.querySelectorAll('.tab-btn[data-tab]').forEach(function (b) {
                b.classList.toggle('active', b.getAttribute('data-tab') === name);
            });
            document.querySelectorAll('.tab-panel').forEach(function (p) {
                p.classList.toggle('active', p.id === 'tab-' + name);
            });
        });
    });

    // Dataset distribution bars (width proportional to the largest train count)
    var bars = document.querySelectorAll('.bar-emotion');
    if (bars.length) {
        var max = 0;
        bars.forEach(function (b) {
            var v = parseInt(b.getAttribute('data-train'), 10) || 0;
            if (v > max) max = v;
        });
        if (max > 0) {
            bars.forEach(function (b) {
                var v = parseInt(b.getAttribute('data-train'), 10) || 0;
                b.style.width = Math.round((v / max) * 100) + '%';
            });
        }
    }
})();
