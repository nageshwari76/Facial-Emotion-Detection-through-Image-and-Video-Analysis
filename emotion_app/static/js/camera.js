/* EmoCare AI — camera capture + image upload + analysis */
(function () {
    'use strict';

    var video = document.getElementById('video');
    var placeholder = document.getElementById('videoPlaceholder');
    var btnStart = document.getElementById('btnStart');
    var btnStop = document.getElementById('btnStop');
    var btnCapture = document.getElementById('btnCapture');
    var imageInput = document.getElementById('imageInput');
    var statusBox = document.getElementById('status');
    var statusText = document.getElementById('statusText');
    var resultEmpty = document.getElementById('resultEmpty');
    var resultBody = document.getElementById('resultBody');
    var capturedWrap = document.getElementById('capturedWrap');
    var capturedImage = document.getElementById('capturedImage');
    var faceCount = document.getElementById('resFaceCount');
    var faceList = document.getElementById('resFaceList');

    var stream = null;
    var capturedUrl = null;
    var analyzing = false;
    var DEFAULT_PLACEHOLDER = 'Click "Start Camera" to begin';

    var EMOJI = {
        angry: '😠', disgust: '🤢', fear: '😨', happy: '😊',
        neutral: '😐', sad: '😢', surprise: '😲', 'class_0': '😶', 'class_1': '😶',
        'class_2': '😶', 'class_3': '😶', 'class_4': '😶', 'class_5': '😶', 'class_6': '😶'
    };

    function csrfToken() {
        var match = document.cookie.match(/csrftoken=([\w-]+)/);
        return match ? match[1] : '';
    }

    function setStatus(show, text) {
        if (statusBox) statusBox.hidden = !show;
        if (statusText) statusText.textContent = text || '';
    }

    function capitalize(s) {
        return s.charAt(0).toUpperCase() + s.slice(1);
    }

    // Show the image that was just captured/uploaded above the prediction.
    function showCaptured(blobOrFile) {
        if (!capturedImage) return;
        if (capturedUrl) URL.revokeObjectURL(capturedUrl);
        capturedUrl = URL.createObjectURL(blobOrFile);
        capturedImage.src = capturedUrl;
        capturedWrap.hidden = false;
    }

    // One row per detected face; numbers match the labels drawn on the image.
    function showFaces(faces, mainId) {
        if (!faceList || !faceCount) return;
        faceList.innerHTML = '';
        faces = faces || [];
        faceCount.hidden = faces.length === 0;
        faceCount.textContent = faces.length === 1 ? '1 face detected'
            : faces.length + ' faces detected — main (largest) face shown below';
        faceList.hidden = faces.length < 2;
        if (faces.length < 2) return;
        faces.forEach(function (f) {
            var li = document.createElement('li');
            if (f.id === mainId) li.className = 'main-face';
            var name = document.createElement('span');
            name.textContent = 'Face ' + f.id + ': ' + (EMOJI[f.emotion] || '😶') + ' ' + capitalize(f.emotion);
            var conf = document.createElement('span');
            conf.className = 'face-conf';
            conf.textContent = f.confidence + '%';
            li.appendChild(name);
            li.appendChild(conf);
            faceList.appendChild(li);
        });
    }

    function showResult(emoji, emotion, conf, message) {
        var confBox = document.getElementById('resConfBox');
        if (confBox) confBox.hidden = false;
        document.getElementById('resEmoji').textContent = emoji;
        document.getElementById('resEmotion').textContent = capitalize(emotion);
        document.getElementById('resConf').textContent = conf + '%';
        document.getElementById('resBar').style.width = Math.min(conf, 100) + '%';
        document.getElementById('resMessage').textContent = message || '';
        resultEmpty.hidden = true;
        resultBody.hidden = false;
    }

    // Invalid input (mobile, plant, chart ...): shown as its own result.
    function showNoFace(label, message) {
        setStatus(false);
        showFaces([]);
        var confBox = document.getElementById('resConfBox');
        if (confBox) confBox.hidden = true;
        document.getElementById('resEmoji').textContent = '🚫';
        document.getElementById('resEmotion').textContent = label || 'No face image detected';
        document.getElementById('resMessage').textContent = message || '';
        resultEmpty.hidden = true;
        resultBody.hidden = false;
    }

    function handleError(msg) {
        setStatus(false);
        resultEmpty.hidden = false;
        resultEmpty.textContent = msg;
        resultBody.hidden = true;
    }

    function cameraErrorText(err) {
        var name = err && err.name;
        if (name === 'NotAllowedError' || name === 'SecurityError') {
            return 'Camera access was blocked. Click the camera icon in the address bar to allow access, then try again.';
        }
        if (name === 'NotFoundError' || name === 'OverconstrainedError') {
            return 'No camera was found on this device. Connect a webcam or use image upload instead.';
        }
        if (name === 'NotReadableError') {
            return 'The camera is already in use by another application. Close it and try again.';
        }
        return 'Could not access the camera. Please allow camera permission or use image upload instead.';
    }

    function showCameraError(msg) {
        // Surface the error where the video should be, not only in the result panel.
        if (placeholder) {
            placeholder.textContent = msg;
            placeholder.hidden = false;
        }
        handleError(msg);
    }

    function sendImage(blobOrFile, isFile) {
        if (analyzing) return;
        analyzing = true;
        setStatus(true, 'Analyzing… (first run loads the AI model — may take a minute)');
        resultEmpty.hidden = false;
        resultEmpty.textContent = 'Analyzing your emotion…';
        resultBody.hidden = true;
        showCaptured(blobOrFile);

        var form = new FormData();
        form.append('image', blobOrFile, isFile ? blobOrFile.name : 'capture.jpg');

        fetch('/analyze/', {
            method: 'POST',
            body: form,
            headers: { 'X-CSRFToken': csrfToken() }
        })
            .then(function (res) {
                // Surface real server errors (403/500 return HTML, not JSON).
                if (!res.ok) {
                    return res.text().then(function (text) {
                        var snippet = (text || '').replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
                        throw new Error('Server error ' + res.status + (snippet ? ': ' + snippet.slice(0, 200) : ''));
                    });
                }
                return res.json();
            })
            .then(function (data) {
                analyzing = false;
                setStatus(false);
                if (data.no_face) {
                    showNoFace(data.label, data.message);
                } else if (data.success) {
                    var emoji = EMOJI[data.emotion] || '😶';
                    // Replace the raw capture with the copy that has every face labeled.
                    if (data.annotated_image && capturedImage) capturedImage.src = data.annotated_image;
                    showResult(emoji, data.emotion, data.confidence, data.message);
                    showFaces(data.faces, data.main_face_id);
                } else {
                    handleError(data.error || 'Something went wrong. Please try again.');
                }
            })
            .catch(function (err) {
                analyzing = false;
                setStatus(false);
                handleError((err && err.message) || 'Network error. Is the Django server running?');
            });
    }

    if (btnStart) {
        btnStart.addEventListener('click', function () {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                showCameraError('Camera not supported in this browser or the page is not secure. ' +
                    'Open the site via https:// or http://localhost and try again.');
                return;
            }
            setStatus(true, 'Starting camera…');
            navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
                .then(function (s) {
                    stream = s;
                    video.srcObject = s;
                    // Ensure the stream is rendered even if autoplay is blocked.
                    var playPromise = video.play();
                    if (playPromise && playPromise.catch) {
                        playPromise.catch(function () { /* muted + playsinline autoplay is allowed */ });
                    }
                    setStatus(false);
                    if (placeholder) {
                        placeholder.textContent = DEFAULT_PLACEHOLDER;
                        placeholder.hidden = true;
                    }
                    btnStart.disabled = true;
                    btnStop.disabled = false;
                    btnCapture.disabled = false;
                })
                .catch(function (err) {
                    showCameraError(cameraErrorText(err));
                });
        });
    }

    if (btnStop) {
        btnStop.addEventListener('click', function () {
            if (stream) {
                stream.getTracks().forEach(function (t) { t.stop(); });
                stream = null;
            }
            video.srcObject = null;
            if (placeholder) {
                placeholder.textContent = DEFAULT_PLACEHOLDER;
                placeholder.hidden = false;
            }
            btnStart.disabled = false;
            btnStop.disabled = true;
            btnCapture.disabled = true;
        });
    }

    if (btnCapture) {
        btnCapture.addEventListener('click', function () {
            if (!stream || analyzing) return;
            // Don't capture before the video has real frames (would send a blank image).
            if (!video.videoWidth || !video.videoHeight) {
                handleError('The camera is still warming up. Wait a moment and try again.');
                return;
            }
            var canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(function (blob) {
                if (blob) sendImage(blob, false);
            }, 'image/jpeg', 0.9);
        });
    }

    if (imageInput) {
        imageInput.addEventListener('change', function () {
            if (imageInput.files && imageInput.files[0]) {
                sendImage(imageInput.files[0], true);
            }
        });

        // Allow dragging & dropping a local image straight onto the upload box.
        var uploadBox = document.querySelector('.upload-box');
        if (uploadBox) {
            uploadBox.addEventListener('dragover', function (e) {
                e.preventDefault();
                uploadBox.classList.add('drag-over');
            });
            uploadBox.addEventListener('dragleave', function () {
                uploadBox.classList.remove('drag-over');
            });
            uploadBox.addEventListener('drop', function (e) {
                e.preventDefault();
                uploadBox.classList.remove('drag-over');
                var file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
                if (file && file.type && file.type.indexOf('image/') === 0) {
                    sendImage(file, true);
                } else {
                    handleError('Please drop an image file (JPG, PNG, etc.).');
                }
            });
        }
    }
})();
