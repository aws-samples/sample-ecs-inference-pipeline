FROM vllm/vllm-openai:v0.26.0 AS base

# Target: linux/amd64

# Model configuration
ENV MODEL_S3_PATH=""
ENV MODEL_NAME=""
ENV SERVED_MODEL_NAME=""
ENV QUANTIZATION="awq"
ENV MAX_SEQ_LEN="2048"
ENV GPU_MEM_UTIL="0.90"
ENV TP_SIZE="1"

# SQS worker configuration
ENV REQUEST_QUEUE_URL=""
ENV DLQ_URL=""
ENV OUTPUT_DESTINATION=""

# Install pinned dependencies for SQS worker (awscli is used by entrypoint.sh for S3 sync)
RUN pip install --no-cache-dir "boto3>=1.35.0,<2" "requests>=2.31.0,<3" "awscli>=1.35.0,<2"

# Worker virtualenv with OpenTelemetry + the CloudWatch plugin (CloudWatch Omni).
# Isolated from the system site-packages so these pins cannot change the
# opentelemetry-* versions vLLM itself depends on.
RUN python3 -m venv /opt/worker-venv && \
    /opt/worker-venv/bin/pip install --no-cache-dir \
      "boto3==1.42.97" \
      "requests==2.32.5" \
      "opentelemetry-distro==0.66b2" \
      "opentelemetry-exporter-otlp==1.45.2" \
      "opentelemetry-instrumentation-botocore==0.66b2" \
      "opentelemetry-instrumentation-requests==0.66b2" \
      "opentelemetry-instrumentation-logging==0.66b2" \
      "cloudwatch-plugin-otel==0.1.1"

# Create a non-root user and set up required directories
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser && \
    mkdir -p /models /app && \
    chown -R appuser:appgroup /models /app

# Copy application files
COPY --chown=appuser:appgroup entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY --chown=appuser:appgroup sqs_worker.py /app/sqs_worker.py
COPY --chown=appuser:appgroup telemetry.py /app/telemetry.py

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
