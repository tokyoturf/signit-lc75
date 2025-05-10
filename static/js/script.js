function updateDetections() {
    fetch('/get_detections')
        .then(response => response.json())
        .then(data => {
            const detectionsDiv = document.getElementById('detections');
            detectionsDiv.innerHTML = '';
            
            if (data.length === 0) {
                detectionsDiv.innerHTML = '<p>No signs detected</p>';
                return;
            }

            data.sort((a, b) => b.confidence - a.confidence);
            
            data.forEach(det => {
                const div = document.createElement('div');
                div.className = 'detection-item';
                div.innerHTML = `
                    <strong>${det.class}</strong>
                    <div>Confidence: ${(det.confidence * 100).toFixed(1)}%</div>
                `;
                detectionsDiv.appendChild(div);
            });
        })
        .catch(error => console.error('Error fetching detections:', error));
}

document.addEventListener('DOMContentLoaded', () => {
    setInterval(updateDetections, 200);
});