// Initialize webcam for all <video> tags on the page
window.onload = function () {
  const videos = document.querySelectorAll('video');

  // Check for browser support
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Your browser does not support webcam access.");
    return;
  }

  // Request webcam once
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
      videos.forEach(video => {
        video.srcObject = stream;
        video.play();
      });
    })
    .catch(error => {
      console.error("Error accessing webcam:", error);
      alert("Unable to access webcam. Please allow camera permissions or try a different browser.");
    });
};

/**
 * Capture image from video element and save to hidden input
 * @param {string} videoId - The ID of the <video> element
 * @param {string} canvasId - The ID of the <canvas> element
 * @param {string} inputId - The ID of the hidden <input> to store base64 image
 * @param {string} [previewId] - Optional: ID of an <img> to preview the snapshot
 */
function takeSnapshot(videoId, canvasId, inputId, previewId) {
  const video = document.getElementById(videoId);
  const canvas = document.getElementById(canvasId);
  const input = document.getElementById(inputId);
  const preview = previewId ? document.getElementById(previewId) : null;

  if (!video || !canvas || !input) {
    console.error("Missing required element(s):", { video, canvas, input });
    alert("Could not capture image. Internal error.");
    return;
  }

  // Set fixed canvas size
  canvas.width = 400;
  canvas.height = 300;

  const context = canvas.getContext('2d');
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Convert to compressed Base64 JPEG
  const base64Image = canvas.toDataURL('image/jpeg', 0.7);
  input.value = base64Image;

  // Optional preview
  if (preview) {
    preview.src = base64Image;
    preview.style.display = 'block';
  }

  alert("Image captured successfully!");
}
