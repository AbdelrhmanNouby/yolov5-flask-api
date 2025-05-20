FROM python:3.10-slim

WORKDIR /app

COPY . /app
COPY yolov5s.pt .
COPY yolov5s.pt /app/yolov5s.pt
COPY yolov5 ./yolov5



RUN apt-get update && \
    apt-get install -y build-essential libgl1-mesa-glx libglib2.0-0 && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
