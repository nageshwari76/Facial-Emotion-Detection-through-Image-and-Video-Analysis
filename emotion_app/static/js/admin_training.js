/* EmoCare AI — admin panel: poll the background model-training job. */
(function () {
    'use strict';

    var btnTrain = document.getElementById('btnTrain');
    var btnEvaluate = document.getElementById('btnEvaluate');
    var progressBox = document.getElementById('trainProgress');
    var messageEl = document.getElementById('trainMessage');
    var barEl = document.getElementById('trainProgressBar');
    var chipEl = document.getElementById('trainStatusChip');
    var modelSelect = document.getElementById('modelSelect');
    var evalModelFile = document.getElementById('evalModelFile');
    var trainModelFile = document.getElementById('trainModelFile');

    // Sync model selector dropdown with hidden form fields.
    if (modelSelect) {
        modelSelect.addEventListener('change', function () {
            var val = modelSelect.value;
            if (evalModelFile) evalModelFile.value = val;
            if (trainModelFile) trainModelFile.value = val;
        });
    }

    // Only poll while the form/status exists (i.e. we are on the training tab page).
    if ((!btnTrain && !btnEvaluate) || !progressBox) return;

    var BUTTON_LABELS = {
        train: '🚀 Run Model',
        evaluate: '⚡ Use Existing h5'
    };

    function setButtons(running) {
        [btnTrain, btnEvaluate].forEach(function (btn, i) {
            if (!btn) return;
            var key = i === 0 ? 'train' : 'evaluate';
            btn.disabled = running;
            if (running) {
                btn.classList.add('btn-processing');
                btn.innerHTML = '<span class="btn-spinner"></span> Processing…';
            } else {
                btn.classList.remove('btn-processing');
                btn.innerHTML = BUTTON_LABELS[key];
            }
        });
    }

    function showProgress(show) {
        if (progressBox) progressBox.hidden = !show;
    }

    function setMessage(text) {
        if (messageEl) messageEl.textContent = text || '';
    }

    function setBar(width) {
        if (barEl) barEl.style.width = Math.max(0, Math.min(100, width || 0)) + '%';
    }

    function setStatusChip(status) {
        if (!chipEl) return;
        chipEl.textContent = (status || 'idle').charAt(0).toUpperCase() + (status || 'idle').slice(1);
        var card = chipEl.closest('.stat-card');
        if (card) {
            card.classList.remove('stat-amber', 'stat-green');
            if (status === 'running') card.classList.add('stat-amber');
            if (status === 'completed') card.classList.add('stat-green');
        }
        setButtons(status === 'running');
    }

    function finished() {
        // Training finished — reload so the fresh graphs / report appear.
        window.location.reload();
    }

    function poll() {
        fetch('/admin-training-status/', { headers: { 'X-Requested-With': 'XMLHttpRequest' } })
            .then(function (res) { return res.json(); })
            .then(function (data) {
                var status = data && data.status;
                if (status === 'running') {
                    showProgress(true);
                    setStatusChip('running');
                    setMessage(data.message || 'Training…');
                    setBar(data.progress || 0);
                    setTimeout(poll, 3000);
                } else {
                    showProgress(false);
                    if (status === 'completed' || status === 'failed') {
                        finished();
                    } else {
                        // idle / not started
                        setStatusChip(status || 'idle');
                        setMessage(data && data.message ? data.message : 'No training run yet.');
                    }
                }
            })
            .catch(function () {
                setTimeout(poll, 5000);
            });
    }

    // Start polling immediately when a run is already in progress, otherwise
    // wait for the user to submit the form (page will reload when it finishes).
    if (chipEl) {
        var current = chipEl.textContent.trim().toLowerCase();
        if (current === 'running') {
            setStatusChip('running');
            showProgress(true);
            setBar(2);
            setTimeout(poll, 500);
        }
    }

    document.querySelectorAll('.train-job-form').forEach(function (form) {
        form.addEventListener('submit', function () {
            showProgress(true);
            setMessage('Starting…');
            setBar(1);
            setStatusChip('running');
            // Poll after the POST completes (redirect back to this page).
            setTimeout(poll, 1500);
        });
    });
})();
