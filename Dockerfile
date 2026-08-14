FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytest

COPY . /app/
ENV PYTHONPATH="/app/src"

# Default to simply providing the inference entrypoint so the evaluator can append paths
ENTRYPOINT ["python", "inference.py"]
CMD ["--help"]
