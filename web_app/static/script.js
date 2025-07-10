document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = document.getElementById('btnText');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    const statusMessage = document.getElementById('statusMessage');
    const resultImage = document.getElementById('resultImage');
    const detectionInfo = document.getElementById('detectionInfo');
    const errorMessage = document.getElementById('errorMessage');

    let selectedFile = null;

    // Click to upload
    uploadArea.addEventListener('click', function() {
        imageInput.click();
    });

    // File selection handler
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            handleFileSelection(file);
        }
    });

    // Drag and drop handlers
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                handleFileSelection(file);
            } else {
                showError('Please select a valid image file.');
            }
        }
    });

    // Analyze button click handler
    analyzeBtn.addEventListener('click', function() {
        if (selectedFile) {
            analyzeImage();
        }
    });

    function handleFileSelection(file) {
        // Validate file type
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
        if (!allowedTypes.includes(file.type)) {
            showError('Invalid file type. Please select a JPG, PNG, GIF, or BMP image.');
            return;
        }

        // Validate file size (16MB max)
        const maxSize = 16 * 1024 * 1024; // 16MB
        if (file.size > maxSize) {
            showError('File size too large. Please select an image smaller than 16MB.');
            return;
        }

        selectedFile = file;
        
        // Update upload area to show selected file
        uploadArea.innerHTML = `
            <div class="upload-content">
                <div class="upload-icon">✅</div>
                <p><strong>Selected:</strong> ${file.name}</p>
                <p class="file-types">Size: ${formatFileSize(file.size)}</p>
                <p class="file-types">Click to select a different image</p>
            </div>
        `;

        // Enable analyze button
        analyzeBtn.disabled = false;
        
        // Hide previous results/errors
        hideResults();
        hideError();
    }

    function analyzeImage() {
        if (!selectedFile) return;

        // Show loading state
        setLoadingState(true);
        hideResults();
        hideError();

        // Create FormData and append the file
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Send request to backend
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            setLoadingState(false);
            
            if (data.success) {
                showResults(data);
            } else {
                showError(data.error || 'An error occurred while processing the image.');
            }
        })
        .catch(error => {
            setLoadingState(false);
            console.error('Error:', error);
            showError('Network error. Please check your connection and try again.');
        });
    }

    function showResults(data) {
        // Count detections with OCR text
        const detectionsWithText = data.detections.filter(d => d.text).length;
        const totalDetections = data.detections.length;
        
        // Update status message
        let message = data.message;
        if (totalDetections > 0 && detectionsWithText > 0) {
            message += ` (${detectionsWithText}/${totalDetections} plates read successfully)`;
        } else if (totalDetections > 0 && detectionsWithText === 0) {
            message += ` (text could not be read)`;
        }
        
        statusMessage.textContent = message;
        statusMessage.className = data.detections.length > 0 ? 'status-message' : 'status-message no-detection';

        // Show result image
        resultImage.src = data.image;
        resultImage.onload = function() {
            resultsSection.style.display = 'block';
        };

        // Show detection info
        if (data.detections.length > 0) {
            displayDetectionInfo(data.detections);
        } else {
            detectionInfo.innerHTML = `
                <div style="text-align: center; padding: 20px; color: #718096;">
                    <p>No license plates were detected in this image.</p>
                    <p style="font-size: 0.9rem; margin-top: 10px;">Try uploading an image with a clearer view of a license plate.</p>
                </div>
            `;
        }
    }

    function displayDetectionInfo(detections) {
        let html = '';
        
        detections.forEach((detection, index) => {
            const [x1, y1, x2, y2] = detection.bbox;
            const width = x2 - x1;
            const height = y2 - y1;
            
            html += `
                <div class="detection-item">
                    <div class="detection-header">
                        Detection ${index + 1}
                        ${detection.text ? `<span class="plate-text">📋 ${detection.text}</span>` : ''}
                    </div>
                    <div class="detection-details">
                        <div><strong>Class:</strong> ${detection.class}</div>
                        <div><strong>Confidence:</strong> ${(detection.confidence * 100).toFixed(1)}%</div>
                        <div><strong>Position:</strong> (${x1}, ${y1})</div>
                        <div><strong>Size:</strong> ${width} × ${height} pixels</div>
                        ${detection.text ? `
                            <div><strong>License Text:</strong> ${detection.text}</div>
                            <div><strong>OCR Method:</strong> ${detection.ocr_method}</div>
                            <div><strong>OCR Confidence:</strong> ${(detection.ocr_confidence * 100).toFixed(1)}%</div>
                        ` : `
                            <div><strong>License Text:</strong> <span style="color: #999;">Could not read</span></div>
                        `}
                    </div>
                </div>
            `;
        });
        
        detectionInfo.innerHTML = html;
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorSection.style.display = 'block';
        resultsSection.style.display = 'none';
    }

    function hideError() {
        errorSection.style.display = 'none';
    }

    function hideResults() {
        resultsSection.style.display = 'none';
    }

    function setLoadingState(loading) {
        if (loading) {
            btnText.textContent = 'Analyzing...';
            loadingSpinner.style.display = 'block';
            analyzeBtn.disabled = true;
        } else {
            btnText.textContent = 'Analyze Image';
            loadingSpinner.style.display = 'none';
            analyzeBtn.disabled = !selectedFile;
        }
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}); 