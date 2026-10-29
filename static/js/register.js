/**
 * Face Registration JavaScript Module
 * Handles browser webcam stream, frame capture, overlay bounding boxes,
 * and automated frame uploads to /api/register_frame.
 */

(() => {
  // DOM Elements
  const video = document.getElementById('webcamVideo');
  const overlayCanvas = document.getElementById('overlayCanvas');
  const captureCanvas = document.getElementById('captureCanvas');
  const cameraPlaceholder = document.getElementById('cameraPlaceholder');
  const nameInput = document.getElementById('personNameInput');
  const startCamBtn = document.getElementById('startCamBtn');
  const stopCamBtn = document.getElementById('stopCamBtn');
  const startCaptureBtn = document.getElementById('startCaptureBtn');
  const stopCaptureBtn = document.getElementById('stopCaptureBtn');
  const progressBar = document.getElementById('progressBar');
  const progressText = document.getElementById('progressCountText');
  const statusAlert = document.getElementById('statusAlert');
  const statusIcon = document.getElementById('statusIcon');
  const statusText = document.getElementById('statusText');
  const completionBanner = document.getElementById('completionBanner');

  const overlayCtx = overlayCanvas.getContext('2d');
  const captureCtx = captureCanvas.getContext('2d');

  let mediaStream = null;
  let isCapturing = false;
  let captureTimeoutId = null;
  const TARGET_COUNT = 50;
  const CAPTURE_INTERVAL_MS = 220; // 4-5 fps capture rate for optimal angle diversity

  // Helper: Update status notification banner
  function setStatus(message, type = 'info', icon = 'ℹ️') {
    statusAlert.className = `alert alert-${type}`;
    statusIcon.textContent = icon;
    statusText.textContent = message;
  }

  // Helper: Update progress bar
  function setProgress(count, max = TARGET_COUNT) {
    const percent = Math.min(100, Math.round((count / max) * 100));
    progressBar.style.width = `${percent}%`;
    progressText.textContent = `${count} / ${max} images`;
  }

  // Draw face box on overlay canvas
  function drawBoundingBox(box) {
    if (!box || box.length !== 4) {
      clearOverlay();
      return;
    }

    const [x, y, w, h] = box;
    const scaleX = overlayCanvas.width / (captureCanvas.width || 640);
    const scaleY = overlayCanvas.height / (captureCanvas.height || 480);

    clearOverlay();
    overlayCtx.lineWidth = 3;
    overlayCtx.strokeStyle = '#10b981';
    overlayCtx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);
  }

  function clearOverlay() {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  }

  // 1. Start browser webcam
  async function startCamera() {
    try {
      setStatus('Requesting webcam access...', 'info', '⏳');
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

      startCamBtn.disabled = true;
      stopCamBtn.disabled = false;
      validateInputs();
      setStatus('Webcam connected. Enter a name and click "Start Auto-Capture".', 'success', '✅');
    } catch (err) {
      console.error('Camera access error:', err);
      let errMsg = 'Could not access webcam. Please verify browser permissions.';
      if (err.name === 'NotAllowedError') {
        errMsg = 'Webcam permission denied. Please allow camera access in your browser address bar.';
      } else if (err.name === 'NotFoundError') {
        errMsg = 'No webcam hardware detected on this device.';
      }
      setStatus(errMsg, 'danger', '❌');
    }
  }

  // 2. Stop camera cleanly
  function stopCamera() {
    stopAutoCapture();

    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
    }

    video.srcObject = null;
    video.style.display = 'none';
    cameraPlaceholder.style.display = 'flex';
    clearOverlay();

    startCamBtn.disabled = false;
    stopCamBtn.disabled = true;
    startCaptureBtn.disabled = true;
    setStatus('Webcam stopped. Click "Start Camera" to resume.', 'info', 'ℹ️');
  }

  // Input validation
  function validateInputs() {
    const rawName = nameInput.value.trim();
    const hasCamera = mediaStream !== null;
    const isValidName = /^[a-zA-Z0-9_\-\. ]{2,50}$/.test(rawName);

    if (hasCamera && isValidName && !isCapturing) {
      startCaptureBtn.disabled = false;
    } else {
      startCaptureBtn.disabled = true;
    }
  }

  nameInput.addEventListener('input', validateInputs);

  // 3. Automated capture loop
  async function captureStep(personName) {
    if (!isCapturing || !mediaStream) return;

    // Grab frame from video
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    const frameData = captureCanvas.toDataURL('image/jpeg', 0.85);

    try {
      const response = await fetch('/api/register_frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: personName, image: frameData }),
      });

      const result = await response.json();

      if (result.success) {
        setProgress(result.count, TARGET_COUNT);
        drawBoundingBox(result.box);
        setStatus(
          `Captured ${result.count} of ${TARGET_COUNT} images. Keep moving your head slowly.`,
          'info',
          '📸'
        );

        if (result.completed || result.count >= TARGET_COUNT) {
          finishRegistration(personName, result.count);
          return;
        }
      } else {
        if (result.error && result.error.includes('No face detected')) {
          setStatus('No face detected. Please face the camera directly in good lighting.', 'warning', '⚠️');
          clearOverlay();
        } else if (result.completed) {
          finishRegistration(personName, result.count);
          return;
        } else {
          setStatus(result.error || result.message || 'Capture error.', 'warning', '⚠️');
        }
      }
    } catch (err) {
      console.error('Frame upload failed:', err);
      setStatus('Network connection error while uploading frame.', 'danger', '❌');
    }

    // Schedule next frame
    if (isCapturing) {
      captureTimeoutId = setTimeout(() => captureStep(personName), CAPTURE_INTERVAL_MS);
    }
  }

  function startAutoCapture() {
    const personName = nameInput.value.trim();
    if (!personName) {
      setStatus('Please enter the person\'s name before capturing.', 'warning', '⚠️');
      nameInput.focus();
      return;
    }

    if (!mediaStream) {
      setStatus('Please start the camera first.', 'warning', '⚠️');
      return;
    }

    isCapturing = true;
    startCaptureBtn.style.display = 'none';
    stopCaptureBtn.style.display = 'block';
    nameInput.disabled = true;
    startCamBtn.disabled = true;
    stopCamBtn.disabled = true;
    completionBanner.style.display = 'none';

    setStatus(`Starting auto-capture for "${personName}". Hold steady...`, 'info', '⚡');
    captureStep(personName);
  }

  function stopAutoCapture() {
    isCapturing = false;
    if (captureTimeoutId) {
      clearTimeout(captureTimeoutId);
      captureTimeoutId = null;
    }

    clearOverlay();
    startCaptureBtn.style.display = 'block';
    stopCaptureBtn.style.display = 'none';
    nameInput.disabled = false;
    startCamBtn.disabled = mediaStream !== null;
    stopCamBtn.disabled = mediaStream === null;
    validateInputs();
  }

  function finishRegistration(personName, count) {
    stopAutoCapture();
    setProgress(count, TARGET_COUNT);
    setStatus(`Successfully registered ${count} face samples for "${personName}"!`, 'success', '🎉');
    completionBanner.style.display = 'flex';
  }

  // Event Listeners
  startCamBtn.addEventListener('click', startCamera);
  stopCamBtn.addEventListener('click', stopCamera);
  startCaptureBtn.addEventListener('click', startAutoCapture);
  stopCaptureBtn.addEventListener('click', () => {
    stopAutoCapture();
    setStatus('Capture paused by user.', 'info', '⏸️');
  });

  // Ensure camera tracks are closed when navigating away
  window.addEventListener('beforeunload', () => {
    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
    }
  });
})();

