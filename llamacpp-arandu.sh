#!/bin/bash -l

#SBATCH --output=slurm_out_%j.txt
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p arandu

PORT=8090
CONTAINER_NAME="llama-srv-${SLURM_JOB_ID}"

echo "Running on"
hostname
date

# Clean up helper function in case the job times out or gets canceled
cleanup() {
    docker logs $CONTAINER_NAME
    echo "Cleaning up Docker container..."
    docker stop $CONTAINER_NAME
}
trap cleanup EXIT

# --- Spin up llama.cpp Server with Auto-Download ---
echo "Starting llama.cpp server and downloading model..."
docker run -d \
    --rm \
    --user "$(id -u):$(id -g)" \
    --name $CONTAINER_NAME \
    --network host \
    --gpus \"device=$CUDA_VISIBLE_DEVICES\" \
    --ipc=host \
    -e HOME=/tmp \
    -e XDG_CACHE_HOME=/tmp/.cache \
    -e LLAMA_CACHE=/tmp/.cache/llama.cpp \
    -e HF_HOME='./cache' \
    -p $PORT:8080 \
    ghcr.io/ggml-org/llama.cpp:server-cuda \
    --hf-repo unsloth/gemma-4-31B-it-GGUF \
    --hf-file BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf \
    --alias "unsloth/gemma-4-31B-it-GGUF" \
    --chat-template-kwargs '{"enable_thinking":true}' \
    --host 0.0.0.0 \
    --port $PORT \
    --n-gpu-layers 99
    
# --- 3. Health Check (Wait for Server to be Ready) ---
echo "Waiting for server to respond..."
SUCCESS=0
for i in {1..1000}; do
    # Check if the container crashed out early
    if [ "$(docker inspect -f '{{.State.Running}}' $CONTAINER_NAME 2>/dev/null)" != "true" ]; then
        echo "CRITICAL: Container exited unexpectedly."
        exit 1
    fi

    # Query HTTP response status code directly rather than raw text matching
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health)
    
    if [ "$HTTP_STATUS" -eq 200 ]; then
        echo "Server is up and verified on port $PORT (HTTP 200)!"
        SUCCESS=1
        break
    fi
    sleep 2
done

if [ $SUCCESS -ne 1 ]; then
    echo "Error: Server failed to start within time."
    exit 1
fi

cp -R /home/[username]/llamacpp-arandu /output/[username]/

# --- Run client code inside the pandas container ---
echo "Running experiment container..."
docker run --rm \
    --network host \
    -v /output/[username]/llamacpp-arandu:/workspace \
    -w /workspace \
    python:3.12-slim \
    bash -c "pip3 install --no-cache-dir pandas==2.2.2 requests==2.34.2 tqdm==4.67.3 && python3 main.py --port $PORT"

mv /output/[username]/llamacpp-arandu/output/* /home/[username]/llamacpp-arandu/output/

rm -r /output/[username]/llamacpp-arandu

echo "Job finished successfully."
date
