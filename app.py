import os
import sys

# Add yolov5 to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov5'))

import torch
from yolov5.models.common import DetectMultiBackend  # Now this works
import cv2
import numpy as np
import base64
from PIL import Image
from flask import Flask, request, jsonify
import torchvision.transforms as transforms

device = torch.device('cpu')
model = DetectMultiBackend('yolov5s.pt', device=device)
model.model.float().eval()

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        file = request.files['image']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Convert PIL to tensor
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        results = model(img_tensor)
        model.names = model.names if hasattr(model, 'names') else model.model.names
        results.render()
        annotated = results.ims[0]
        annotated_img = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
        _, img_encoded = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

        return jsonify({
            'status': 'success',
            'image_base64': img_base64,
            'labels': results.pandas().xyxy[0].to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
