#!/bin/bash
set -e

# Graceful shutdown handler
cleanup() {
    echo "Received shutdown signal, cleaning up..."
    if [ -n "$WORKER_PID" ]; then
        kill "$WORKER_PID" 2>/dev/null || true
    fi
    if [ -n "$VLLM_PID" ]; then
        kill "$VLLM_PID" 2>/dev/null || true
    fi
    wait
    exit 0
}

trap cleanup SIGTERM SIGINT SIGQUIT

# --- Step 1: Download model from S3 or use HuggingFace ---
if [ -n "${MODEL_S3_PATH}" ] && [ "${MODEL_S3_PATH}" != "none" ] && [ "${MODEL_S3_PATH}" != "s3://none/none/" ]; then
    echo "Downloading model from ${MODEL_S3_PATH} to /models/${MODEL_NAME}..."
    if ! aws s3 sync "${MODEL_S3_PATH}" "/models/${MODEL_NAME}" --quiet --no-progress; then
        echo "ERROR: Failed to download model from S3 path: ${MODEL_S3_PATH}"
        exit 1
    fi
    echo "Model download complete."
    MODEL_PATH="/models/${MODEL_NAME}"
else
    echo "No S3 path specified, vLLM will download ${MODEL_NAME} from HuggingFace."
    MODEL_PATH="${MODEL_NAME}"
fi

# --- Step 2: Start vLLM server ---
VLLM_ARGS=(
    "--model" "${MODEL_PATH}"
    "--served-model-name" "${SERVED_MODEL_NAME:-${MODEL_NAME}}"
    "--tensor-parallel-size" "${TP_SIZE:-1}"
    "--max-model-len" "${MAX_SEQ_LEN}"
    "--gpu-memory-utilization" "${GPU_MEM_UTIL:-0.90}"
    "--host" "0.0.0.0"
    "--port" "8000"
)

if [ "${QUANTIZATION}" != "none" ]; then
    VLLM_ARGS+=("--quantization" "${QUANTIZATION}")
fi

echo "Starting vLLM server with args: ${VLLM_ARGS[*]}"
python3 -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" &
VLLM_PID=$!

# --- Step 3: Wait for vLLM to be ready ---
echo "Waiting for vLLM server to be ready..."
MAX_RETRIES=120
RETRY_INTERVAL=5
for i in $(seq 1 $MAX_RETRIES); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server is ready."
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server process exited unexpectedly."
        exit 1
    fi
    echo "vLLM not ready yet (attempt ${i}/${MAX_RETRIES}), retrying in ${RETRY_INTERVAL}s..."
    sleep $RETRY_INTERVAL
done

if ! curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to become ready after $((MAX_RETRIES * RETRY_INTERVAL))s."
    exit 1
fi

# --- Step 4: Start SQS worker ---
echo "Starting SQS worker..."
python3 /app/sqs_worker.py &
WORKER_PID=$!

# Wait for either process to exit
wait -n "$VLLM_PID" "$WORKER_PID"
EXIT_CODE=$?

echo "A process exited with code ${EXIT_CODE}, shutting down..."
cleanup
