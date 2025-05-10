import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov5')))

import cv2
import torch
from flask import Flask, render_template, Response, jsonify, url_for
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes

app = Flask(__name__)

# Load Model YOLOv5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(weights='best-windows-2.pt', device=device)

CLASS_NAMES = ['Y', 'E', 'O', 'F', 'P', 'Z', 'G', 'Q', 'Halo', 'H', 'R', 'NamaAku', 
               'I', 'S', 'J', 'T', 'A', 'K', 'U', 'B', 'L', 'V', 'C', 'M', 'W', 'D', 'N', 'X']

current_detections = []

def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    return img, img_tensor

def process_detections(pred, frame_shape):
    global current_detections
    current_detections = []
    
    for det in pred:
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                class_name = CLASS_NAMES[int(cls)]
                current_detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'box': [int(x) for x in xyxy]
                })
    return current_detections

def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    img = torch.zeros((1, 3, 640, 640), device=device)
    model(img)

    while True:
        success, frame = cap.read()
        if not success:
            break

        img, img_tensor = preprocess_frame(frame)

        with torch.no_grad():
            pred = model(img_tensor)

        pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45)

        process_detections(pred, frame.shape)

        for det in current_detections:
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html', 
                         class_names=CLASS_NAMES,
                         css_url=url_for('static', filename='css/styles.css'),
                         js_url=url_for('static', filename='js/script.js'))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detections')
def get_detections():
    return jsonify(current_detections)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)