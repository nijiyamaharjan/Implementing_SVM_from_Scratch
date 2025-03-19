const canvas = document.getElementById('drawing-canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;

ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = 'white';
ctx.lineWidth = 15;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
}

function draw(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function clearCanvas() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    hideResult();
}

function switchTab(tab) {
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabs.forEach(t => t.classList.remove('active'));
    tabContents.forEach(c => c.classList.remove('active'));
    
    if (tab === 'draw') {
        document.querySelector('.tab:first-child').classList.add('active');
        document.getElementById('draw-tab').classList.add('active');
    } else {
        document.querySelector('.tab:last-child').classList.add('active');
        document.getElementById('upload-tab').classList.add('active');
    }
    
    hideResult();
}

function previewImage() {
    const fileInput = document.getElementById('image-upload');
    const previewImage = document.getElementById('preview-image');
    
    if (fileInput.files && fileInput.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewImage.classList.remove('hidden');
        }
        reader.readAsDataURL(fileInput.files[0]);
        hideResult();
    }
}

function predictDrawing() {
    showLoading();
    const imageData = canvas.toDataURL('image/png');
    
    const formData = new FormData();
    formData.append('drawing', imageData);
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        showResult(data.prediction);
    })
    .catch(error => {
        hideLoading();
        console.error('Error:', error);
        alert('An error occurred while predicting the digit.');
    });
}

function predictUploadedImage() {
    const fileInput = document.getElementById('image-upload');
    
    if (fileInput.files && fileInput.files[0]) {
        showLoading();
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            showResult(data.prediction);
        })
        .catch(error => {
            hideLoading();
            console.error('Error:', error);
            alert('An error occurred while predicting the digit.');
        });
    } else {
        alert('Please select an image first.');
    }
}

function showResult(prediction) {
    document.getElementById('result').classList.remove('hidden');
    document.getElementById('predicted-digit').textContent = prediction;
}

function hideResult() {
    document.getElementById('result').classList.add('hidden');
}

function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

canvas.addEventListener('touchstart', function(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousedown', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
});

canvas.addEventListener('touchmove', function(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousemove', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
});

canvas.addEventListener('touchend', function(e) {
    e.preventDefault();
    const mouseEvent = new MouseEvent('mouseup', {});
    canvas.dispatchEvent(mouseEvent);
});
