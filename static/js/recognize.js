/**
 * Live Face Recognition & Biometric Login JavaScript Module
 * Handles browser camera stream, overlay canvas rendering,
 * and periodic AJAX requests to /api/recognize.
 */

(() => {
  // DOM Elements
  const video = document.getElementById('webcamVideo');
  const overlayCanvas = document.getElementById('overlayCanvas');
  const captureCanvas = document.getElementById('captureCanvas');
  const cameraPlaceholder = document.getElementById('cameraPlaceholder');
  const startRecBtn = document.getElementById('startRecBtn');
  const stopRecBtn = document.getElementById('stopRecBtn');
  const thresholdSlider = document.getElementById('thresholdSlider');
  const thresholdValueText = document.getElementById('thresholdValueText');
  const unloadedModelAlert = document.getElementById('unloadedModelAlert');
  const cameraStage = document.getElementById('cameraStageDiagnostic');

  // Diagnostics & Status
  const matchStatusIcon = document.getElementById('matchStatusIcon');
  const matchName = document.getElementById('matchName');
  const matchSubtext = document.getElementById('matchSubtext');
  const matchBadgeContainer = document.getElementById('matchBadgeContainer');
  const diagFaceCount = document.getElementById('diagFaceCount');
  const diagScore = document.getElementById('diagScore');
  const diagRegisteredCount = document.getElementById('diagRegisteredCount');

  const overlayCtx = overlayCanvas.getContext('2d');
  const captureCtx = captureCanvas.getContext('2d');

  let mediaStream = null;
  let isRecognizing = false;
  let isRequestPending = false;
  let recognitionTimeoutId = null;
  const INFERENCE_INTERVAL_MS = 200;

  // Icons SVG dictionary
  const ICONS = {
    user: `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`,
    verified: `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="#10b981" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>`,
    unknown: `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="#f43f5e" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`,
    scanning: `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="#38bdf8" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`,
    warning: `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="#f59e0b" stroke-width="2"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`
  };

  thresholdSlider.addEventListener('input', () => {
    thresholdValueText.textContent = parseFloat(thresholdSlider.value).toFixed(1);
  });

  async function checkInitialModelStatus() {
    try {
      const res = await fetch('/api/stats');
      const stats = await res.json();
      diagRegisteredCount.textContent = stats.user_count;
      if (!stats.model_active && !stats.model_exists) {
        unloadedModelAlert.style.display = 'flex';
      }
    } catch (err) {
      console.error('Failed to query model stats:', err);
    }
  }

  async function startRecognitionCamera() {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });

      video.srcObject = mediaStream;
      video.style.display = 'block';
      cameraPlaceholder.style.display = 'none';
      if (cameraStage) cameraStage.classList.add('active');

      video.onloadedmetadata = () => {
        overlayCanvas.width = video.videoWidth || 640;
        overlayCanvas.height = video.videoHeight || 480;
        captureCanvas.width = video.videoWidth || 640;
        captureCanvas.height = video.videoHeight || 480;
      };

      startRecBtn.disabled = true;
      stopRecBtn.disabled = false;
      isRecognizing = true;
      updateStatusCard('Scanning for faces...', 'Align your face in the camera frame', 'scanning', 'info', 'Scanning');
      
      recognizeStep();
    } catch (err) {
      console.error('Camera access error:', err);
      let errMsg = 'Could not access camera.';
      if (err.name === 'NotAllowedError') {
        errMsg = 'Camera permission denied. Allow camera permissions in your browser.';
      }
      alert(errMsg);
    }
  }

  function stopRecognitionCamera() {
    isRecognizing = false;
    if (recognitionTimeoutId) {
      clearTimeout(recognitionTimeoutId);
      recognitionTimeoutId = null;
    }

    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
    }

    video.srcObject = null;
    video.style.display = 'none';
    cameraPlaceholder.style.display = 'flex';
    if (cameraStage) cameraStage.classList.remove('active');
    clearOverlay();

    startRecBtn.disabled = false;
    stopRecBtn.disabled = true;
    updateStatusCard('Camera Inactive', 'Click "Start Camera" to begin', 'user', 'info', 'Idle');
    diagFaceCount.textContent = '0';
    diagScore.textContent = '--';
  }

  function renderFaceBoxes(faces) {
    clearOverlay();
    if (!faces || faces.length === 0) return;

    const scaleX = overlayCanvas.width / (captureCanvas.width || 640);
    const scaleY = overlayCanvas.height / (captureCanvas.height || 480);

    faces.forEach((face) => {
      const [x, y, w, h] = face.box;
      const rx = x * scaleX;
      const ry = y * scaleY;
      const rw = w * scaleX;
      const rh = h * scaleY;

      const isMatch = face.matched;
      const boxColor = isMatch ? '#10b981' : '#f43f5e';
      const labelText = isMatch 
        ? `VERIFIED: ${face.name.toUpperCase()} (DIST: ${face.confidence})`
        : `UNKNOWN (DIST: ${face.confidence})`;

      // Cyber Corner Brackets
      overlayCtx.lineWidth = 2.5;
      overlayCtx.strokeStyle = boxColor;

      const cornerLen = Math.min(rw, rh) * 0.25;
      overlayCtx.beginPath();
      overlayCtx.moveTo(rx, ry + cornerLen);
      overlayCtx.lineTo(rx, ry);
      overlayCtx.lineTo(rx + cornerLen, ry);

      overlayCtx.moveTo(rx + rw - cornerLen, ry);
      overlayCtx.lineTo(rx + rw, ry);
      overlayCtx.lineTo(rx + rw, ry + cornerLen);

      overlayCtx.moveTo(rx + rw, ry + rh - cornerLen);
      overlayCtx.lineTo(rx + rw, ry + rh);
      overlayCtx.lineTo(rx + rw - cornerLen, ry + rh);

      overlayCtx.moveTo(rx + cornerLen, ry + rh);
      overlayCtx.lineTo(rx, ry + rh);
      overlayCtx.lineTo(rx, ry + rh - cornerLen);
      overlayCtx.stroke();

      // Label background tag
      overlayCtx.font = '600 12px "Plus Jakarta Sans", sans-serif';
      const textWidth = overlayCtx.measureText(labelText).width;
      const tagHeight = 22;
      const tagY = ry > tagHeight + 6 ? ry - tagHeight - 4 : ry + rh + 4;

      overlayCtx.fillStyle = boxColor;
      overlayCtx.fillRect(rx, tagY, textWidth + 14, tagHeight);

      // Label text
      overlayCtx.fillStyle = '#ffffff';
      overlayCtx.fillText(labelText, rx + 7, tagY + 15);
    });
  }

  function clearOverlay() {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  }

  function updateStatusCard(name, subtext, iconKey, badgeType, badgeLabel) {
    matchName.textContent = name;
    matchSubtext.textContent = subtext;
    matchStatusIcon.innerHTML = ICONS[iconKey] || ICONS.user;
    matchBadgeContainer.innerHTML = `<span class="badge badge-${badgeType}">${badgeLabel}</span>`;
  }

  async function recognizeStep() {
    if (!isRecognizing || !mediaStream) return;

    if (!isRequestPending && video.readyState >= 2) {
      isRequestPending = true;

      captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
      const frameData = captureCanvas.toDataURL('image/jpeg', 0.8);
      const currentThreshold = parseFloat(thresholdSlider.value) || 80.0;

      try {
        const response = await fetch('/api/recognize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: frameData, threshold: currentThreshold }),
        });

        const result = await response.json();

        if (result.success) {
          renderFaceBoxes(result.faces);
          diagFaceCount.textContent = result.faces.length;

          if (result.faces.length > 0) {
            const primary = result.faces[0];
            diagScore.textContent = primary.confidence;

            if (primary.matched) {
              updateStatusCard(
                `Welcome, ${primary.name}!`,
                `Identity Verified (Distance: ${primary.confidence} < ${currentThreshold})`,
                'verified',
                'success',
                'Verified'
              );
            } else {
              updateStatusCard(
                'Unregistered Face',
                `Access Denied (Distance: ${primary.confidence} ≥ ${currentThreshold})`,
                'unknown',
                'danger',
                'Unknown'
              );
            }
          } else {
            diagScore.textContent = '--';
            updateStatusCard('Scanning for faces...', 'No face currently detected in frame', 'scanning', 'warning', 'No Face');
          }
        } else if (result.model_loaded === false) {
          unloadedModelAlert.style.display = 'flex';
          updateStatusCard('Model Not Trained', 'Train the model first to recognize users', 'warning', 'danger', 'No Model');
        }
      } catch (err) {
        console.error('Recognition API error:', err);
      } finally {
        isRequestPending = false;
      }
    }

    if (isRecognizing) {
      recognitionTimeoutId = setTimeout(recognizeStep, INFERENCE_INTERVAL_MS);
    }
  }

  startRecBtn.addEventListener('click', startRecognitionCamera);
  stopRecBtn.addEventListener('click', stopRecognitionCamera);

  window.addEventListener('beforeunload', () => {
    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
    }
  });

  checkInitialModelStatus();
})();
