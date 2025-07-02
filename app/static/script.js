document.addEventListener('DOMContentLoaded', () => {
  // Ruota il logo al passaggio del mouse
  const logo = document.getElementById("logo");
  logo.addEventListener("mouseenter", () => {
    logo.style.transform = "rotate(360deg)";
  });
  logo.addEventListener("mouseleave", () => {
    logo.style.transform = "rotate(0deg)";
  });

  if (window.location.pathname !== "/")
    return;

  // --- DOM Element References ---
  const uploadSection = document.getElementById('uploadSection');
  const imageUpload = document.getElementById('imageUpload');
  const imagePreview = document.getElementById('imagePreview');
  const uploadHelp = document.getElementById('uploadHelp');
  const classifyBtn = document.getElementById('classifyBtn');
  const initialState = document.getElementById('initialState');
  const loadingState = document.getElementById('loadingState');
  const resultState = document.getElementById('resultState');
  const explainCheck = document.getElementById('explainCheck');
  const gradcamState = document.getElementById('gradcamState');
  const gradcamImage = document.getElementById('gradcamImage');
  const feedbackSection = document.getElementById('feedbackSection');
  const feedbackYesBtn = document.getElementById('feedbackYesBtn');
  const feedbackNoBtn = document.getElementById('feedbackNoBtn');

  // Bootstrap Toast Elements
  const liveToast = document.getElementById('liveToast');
  const toastBody = liveToast.querySelector('.toast-body');
  const toast = new bootstrap.Toast(liveToast);

  let uploadedFile = null;
  let currentImageId = null;

  // --- Functions ---

  /**
   * Prevents default behavior for drag-and-drop events.
   * @param {Event} e - The event object.
   */
  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  /**
   * Handles the files provided by the user (from click or drag-drop).
   * @param {FileList} files - The list of files to process.
   */
  function handleFiles(files) {
    const file = files[0];
    if (file) {
      // Check file type
      const allowedTypes = ['image/jpeg', 'image/png'];
      if (!allowedTypes.includes(file.type)) {
        showAlert('Invalid file type. Please upload a JPG or PNG image.', 'error');
        return;
      }

      // Check file size
      const maxSize = 5 * 1024 * 1024; // 5MB
      if (file.size > maxSize) {
        showAlert('File is too large. Maximum size is 5MB.', 'error');
        return;
      }

      uploadedFile = file;
      const reader = new FileReader();
      reader.onload = function(e) {
        imagePreview.src = e.target.result;
        imagePreview.style.display = 'block';
        uploadHelp.textContent = `File: ${uploadedFile.name}`;
        uploadSection.classList.add('has-image');
        classifyBtn.disabled = false;
      }
      reader.readAsDataURL(uploadedFile);
      resetResultState();
    }
  }

  /**
   * Resets the result section to its initial placeholder state.
   */
  function resetResultState() {
    initialState.style.display = 'block';
    resultState.style.display = 'none';
    loadingState.style.display = 'none';
    gradcamState.style.display = 'none';
    feedbackSection.style.display = 'none';
    resultState.innerHTML = '';
    gradcamImage.src = ''; // Reset Grad-CAM image
  }

  /**
   * Displays the classification results with progress bars.
   * @param {string} id - The ID of the classified image.
   * @param {number} real - The percentage for 'Real'.
   * @param {number} fake - The percentage for 'Fake'.
   * @param {string} [gradcamPath] - Optional path to the Grad-CAM image.
   */
  function displayResults(id, real, fake, gradcamPath) {
    currentImageId = id;
    resultState.style.display = 'block';
    resultState.innerHTML = `
      <h4 class="mb-4 text-center">Analysis Complete</h4>

      <div class="mb-4 w-100">
        <div class="progress-label">
          <span><i class="fa-solid fa-circle-check text-success me-2"></i>Real</span>
          <span>${real}%</span>
        </div>
        <div class="progress" style="height: 1.5rem;">
          <div class="progress-bar bg-success" role="progressbar" style="width: ${real}%" aria-valuenow="${real}" aria-valuemin="0" aria-valuemax="100"></div>
        </div>
      </div>

      <div class="w-100">
        <div class="progress-label">
          <span><i class="fa-solid fa-circle-xmark text-danger me-2"></i>Fake</span>
          <span>${fake}%</span>
        </div>
        <div class="progress" style="height: 1.5rem;">
          <div class="progress-bar bg-danger" role="progressbar" style="width: ${fake}%" aria-valuenow="${fake}" aria-valuemin="0" aria-valuemax="100"></div>
        </div>
      </div>
    `;

    if (gradcamPath) {
      gradcamImage.src = gradcamPath;
      gradcamState.style.display = 'flex';
    } else {
      gradcamState.style.display = 'none';
    }

    feedbackSection.style.display = 'block';

    if (real + fake !== 100.0) {
      showAlert('WUT?');
    }
  }

  function showAlert(message, type = 'info') {
    if (type === 'success') {
      liveToast.classList.add('text-bg-success');
      liveToast.classList.remove('text-bg-primary', 'text-bg-danger');
    } else if (type === 'error') {
      liveToast.classList.add('text-bg-danger');
      liveToast.classList.remove('text-bg-primary', 'text-bg-success');
    } else {
      liveToast.classList.add('text-bg-primary');
      liveToast.classList.remove('text-bg-success', 'text-bg-danger');
    }

    toastBody.textContent = message;
    toast.show();
  }

  /**
   * Sends feedback to the server.
   * @param {boolean} isCorrect - Whether the prediction was correct.
   */
  async function sendFeedback(isCorrect) {
    if (!currentImageId) return;

    try {
      const response = await fetch('/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          id: currentImageId,
          correct: isCorrect
        }),
      });

      if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
      const json = await response.json();
      if (json.error) throw new Error(json.error);

      showAlert('Thank you for your feedback!');
    } catch (err) {
      console.error('Feedback Error:', err);
      showAlert(`Could not send feedback: ${err.message}`, 'error');
    } finally {
      feedbackSection.style.display = 'none';
    }
  }

  // Setup drag and drop listeners.
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadSection.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
  });
  ['dragenter', 'dragover'].forEach(eventName => {
    uploadSection.addEventListener(eventName, () => uploadSection.classList.add('dragover'), false);
  });
  ['dragleave', 'drop'].forEach(eventName => {
    uploadSection.addEventListener(eventName, () => uploadSection.classList.remove('dragover'), false);
  });
  uploadSection.addEventListener('drop', (e) => handleFiles(e.dataTransfer.files), false);

  // Trigger file input when the upload section is clicked
  uploadSection.addEventListener('click', () => {
    imageUpload.click();
  });

  // Handle file selection from the file input
  imageUpload.addEventListener('change', (event) => {
    handleFiles(event.target.files);
  });

  // Handle classify button click to send data to Flask
  classifyBtn.addEventListener('click', async () => {
    if (!uploadedFile) {
      showAlert('Please upload an image first!', 'error');
      return;
    }

    // Play music and record start time
    const audio = new Audio('/static/trinita.mp3');
    const startTime = Date.now();
    audio.play();

    // Show loading state and disable button
    initialState.style.display = 'none';
    resultState.style.display = 'none';
    gradcamState.style.display = 'none';
    feedbackSection.style.display = 'none';
    loadingState.style.display = 'block';
    classifyBtn.disabled = true;

    const formData = new FormData();
    formData.append('image', uploadedFile);
    formData.append('explain', explainCheck.checked);

    let resultData, fetchError;
    try {
      const response = await fetch('/api/classify', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
      const json = await response.json();
      if (json.error) throw new Error(json.error);
      resultData = json;
    } catch (err) {
      fetchError = err;
    }

    // Ensure at least 8s of loading
    const elapsed = Date.now() - startTime;
    if (elapsed < 8000) {
      await new Promise(res => setTimeout(res, 8000 - elapsed));
    }

    // Stop audio and reset
    audio.pause();
    audio.currentTime = 0;

    // Hide spinner and re-enable button
    loadingState.style.display = 'none';
    classifyBtn.disabled = false;

    if (fetchError) {
      console.error('Classification Error:', fetchError);
      showAlert(`An error occurred: ${fetchError.message}`, 'error');
      resetResultState();
    } else {
      displayResults(resultData.id, resultData.detector.real, resultData.detector.fake, resultData.gradcam_image_path);
    }
  });

  feedbackYesBtn.addEventListener('click', () => sendFeedback(true));
  feedbackNoBtn.addEventListener('click', () => sendFeedback(false));
});
