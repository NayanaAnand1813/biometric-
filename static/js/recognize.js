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
  const INFERENCE_INTERVAL_MS = 200; // ~5 requests per second for smooth AI response

  // Threshold Slider
  thresholdSlider.addEventListener('input', () => {
    thresholdValueText.textContent = parseFloat(thresholdSlider.value).toFixed(1);
  });

  // Check model status on page load
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

  // 1. Start browser webcam
  async function startRecognitionCamera() {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
        audio: false,
      });

      video.srcObject = mediaStream;
      video.style.display = 'block';
      cameraPlaceholder.style.display = 'none';

      video.onloadedmetadata = () => {
        overlayCanvas.width = video.videoWidth || 640;
        overlayCanvas.height = video.videoHeight || 480;
        captureCanvas.width = video.videoWidth || 640;
        captureCanvas.height = video.videoHeight || 480;
      };

      startRecBtn.disabled = true;
      stopRecBtn.disabled = false;
      isRecognizing = true;
      updateStatusCard('Scanning for faces...', 'Align your face in the camera frame', '👤', 'info', 'Scanning');
      
      // Start recognition loop
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

  // 2. Stop camera cleanly
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
    clearOverlay();

    startRecBtn.disabled = false;
    stopRecBtn.disabled = true;
    updateStatusCard('Camera Inactive', 'Click "Start Recognition" to begin', '👤', 'info', 'Idle');
    diagFaceCount.textContent = '0';
    diagScore.textContent = '--';
  }

  // Draw bounding boxes on overlay canvas
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
      const boxColor = isMatch ? '#10b981' : '#ef4444';
      const labelText = isMatch 
        ? `${face.name} (${face.confidence})`
        : `Unknown (${face.confidence})`;

      // Draw box
      overlayCtx.lineWidth = 3;
      overlayCtx.strokeStyle = boxColor;
      overlayCtx.strokeRect(rx, ry, rw, rh);

      // Draw label background tag
      overlayCtx.font = 'bold 15px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      const textWidth = overlayCtx.measureText(labelText).width;
      const tagHeight = 24;
      const tagY = ry > tagHeight + 6 ? ry - tagHeight - 4 : ry + rh + 4;

      overlayCtx.fillStyle = boxColor;
      overlayCtx.fillRect(rx, tagY, textWidth + 14, tagHeight);

      // Label text
      overlayCtx.fillStyle = '#ffffff';
      overlayCtx.fillText(labelText, rx + 7, tagY + 17);
    });
  }

  function clearOverlay() {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  }

  function updateStatusCard(name, subtext, icon, badgeType, badgeLabel) {
    matchName.textContent = name;
    matchSubtext.textContent = subtext;
    matchStatusIcon.textContent = icon;
    matchBadgeContainer.innerHTML = `<span class="badge badge-${badgeType}">${badgeLabel}</span>`;
  }

  // 3. Periodic Inference Loop with Overlap Prevention
  async function recognizeStep() {
    if (!isRecognizing || !mediaStream) return;

    if (!isRequestPending && video.readyState >= 2) {
      isRequestPending = true;

      // Draw current frame to hidden capture canvas
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
                '🔓',
                'success',
                'Verified'
              );
            } else {
              updateStatusCard(
                'Unregistered Face',
                `Access Denied (Distance: ${primary.confidence} ≥ ${currentThreshold})`,
                '🚫',
                'danger',
                'Unknown'
              );
            }
          } else {
            diagScore.textContent = '--';
            updateStatusCard('Scanning for faces...', 'No face currently detected in frame', '🔍', 'warning', 'No Face');
          }
        } else if (result.model_loaded === false) {
          unloadedModelAlert.style.display = 'flex';
          updateStatusCard('Model Not Trained', 'Train the model first to recognize users', '⚠️', 'danger', 'No Model');
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

  // Event Listeners
  startRecBtn.addEventListener('click', startRecognitionCamera);
  stopRecBtn.addEventListener('click', stopRecognitionCamera);

  window.addEventListener('beforeunload', () => {
    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
    }
  });

  checkInitialModelStatus();
})();

