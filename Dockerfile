FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

# xgboost needs the OpenMP runtime, which is missing in the slim image
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ data/
COPY train.py predict.py ./

# train the model at build time -> the image is fully self-contained and reproducible
RUN python train.py

EXPOSE 8000
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
