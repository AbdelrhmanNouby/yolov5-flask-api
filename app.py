import torch
import cv2
import numpy as np
import base64
import time
from PIL import Image
from flask import Flask, request, jsonify

device = torch.device('cpu')  # Change to 'cuda' if GPU is available
print(f"🚀 Using device: {device}")

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to(device)
model.eval()

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    return 'pong', 200

@app.route('/detect', methods=['POST'])
def detect():
    try:
        file = request.files['image']
        img_bytes = file.read()

        # Decode image
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Start processing timer
        start_time = time.time()

        # Run YOLOv5 detection
        results = model(pil_img)
        results.render()

        processing_latency = (time.time() - start_time) * 1000  # in milliseconds

        # Encode result image
        rendered = results.ims[0]
        annotated_img = cv2.cvtColor(np.array(rendered), cv2.COLOR_RGB2BGR)
        _, img_encoded = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

        return jsonify({
            'status': 'success',
            'image_base64': img_base64,
            'labels': results.pandas().xyxy[0].to_dict(orient="records"),
            'processing_latency_ms': round(processing_latency, 2)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
