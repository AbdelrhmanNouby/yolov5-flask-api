import torch
import cv2
import numpy as np
import base64
import time
from PIL import Image
import asyncio
import websockets
import json

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to(torch.device('cpu'))
model.eval()

async def detect(websocket, path):
    async for message in websocket:
        try:
            data = json.loads(message)
            img_base64 = data['image_base64']
            img_bytes = base64.b64decode(img_base64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            start_time = time.time()
            results = model(pil_img)
            results.render()
            rendered = results.ims[0]
            annotated_img = cv2.cvtColor(np.array(rendered), cv2.COLOR_RGB2BGR)
            _, img_encoded = cv2.imencode('.jpg', annotated_img)
            img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
            processing_latency = (time.time() - start_time) * 1000
            response = {
                'status': 'success',
                'image_base64': img_base64,
                'labels': results.pandas().xyxy[0].to_dict(orient="records"),
                'processing_latency_ms': round(processing_latency, 2)
            }
        except Exception as e:
            response = {'status': 'error', 'message': str(e)}
        await websocket.send(json.dumps(response))


async def main():
    async with websockets.serve(detect, "0.0.0.0", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
