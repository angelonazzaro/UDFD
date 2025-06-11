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
  const resultSection = document.getElementById('resultSection');
  const initialState = document.getElementById('initialState');
  const loadingState = document.getElementById('loadingState');
  const resultState = document.getElementById('resultState');
  const uploadCheck = document.getElementById('uploadCheck');

  // Bootstrap Toast Elements
  const liveToast = document.getElementById('liveToast');
  const toastBody = liveToast.querySelector('.toast-body');
  const toast = new bootstrap.Toast(liveToast);

  let uploadedFile = null;

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
    resultState.innerHTML = '';
  }

  /**
   * Displays the classification results with progress bars.
   * @param {number} real - The percentage for 'Real'.
   * @param {number} fake - The percentage for 'Fake'.
   */
  function displayResults(real, fake) {
    resultState.style.display = 'block';
    const resultHTML = `
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
    resultState.innerHTML = resultHTML;
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

    // Show loading state and disable button
    initialState.style.display = 'none';
    resultState.style.display = 'none';
    loadingState.style.display = 'block';
    classifyBtn.disabled = true;

    // Prepare form data to send
    const formData = new FormData();
    formData.append('image', uploadedFile);
    formData.append('upload', uploadCheck.checked);

    try {
      // Send the image to the Flask backend
      const response = await fetch('/api/classify', {
        method: 'POST',
        body: formData,
      });
      console.log(response)

      if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`);
      }

      const data = await response.json();
      console.log(data)
      if(data.error){
        throw new Error(data.error);
      }

      displayResults(data.real, data.fake);
      if (uploadCheck.checked) {
        showAlert('Image uploaded successfully!', 'success');
      }

    } catch (error) {
      console.error('Classification Error:', error);
      showAlert(`An error occurred: ${error.message}`, 'error');
      resetResultState();
    } finally {
      // Hide loading state and re-enable button
      loadingState.style.display = 'none';
      classifyBtn.disabled = false;
    }
  });
});
