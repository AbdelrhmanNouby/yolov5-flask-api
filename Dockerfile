FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y build-essential libgl1-mesa-glx libglib2.0-0 && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Use gunicorn to run the Flask app. It will serve on port 5000 inside the container.
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]