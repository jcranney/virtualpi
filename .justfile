universe *FLAGS:
    source ./keys/universe.keys && ipython {{FLAGS}} virtualpi.py pdfs/universe

scan *FLAGS: activate
    source ./.env && ipython {{FLAGS}} scan_messages.py

activate:
    source ./venv/bin/activate

clean:
    rm -f ./pdfs/docs.pkl

setup:
    rm -rf ./venv
    python -m venv venv
    just activate
    pip install -r requirements.txt
    mkdir -p pdfs

llama:
    ./llama.cpp/server -m ~/llama-tests/model/ggml-model-f32.gguf --gpu-layers 35 -c 2048 --host 0.0.0.0 --port 8002 --embeddings

py:
    echo $(which ipython)
