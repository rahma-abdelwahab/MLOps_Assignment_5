FROM python:3.10-slim

ARG RUN_ID

RUN echo "Downloading model for Run ID: ${RUN_ID}"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train.py"]
