FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytest

COPY . /app/
ENV PYTHONPATH="/app/src"

CMD ["python", "inference.py", "evaluation_dataset_stress/eval/sample_000/reference.png", "evaluation_dataset_stress/eval/sample_000/search.png"]
