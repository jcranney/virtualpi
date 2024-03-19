scan *FLAGS: activate
    source ./.env && ipython {{FLAGS}} scan_messages.py

activate:
    source ./venv/bin/activate

clean:
    rm -f ./pdfs/data.pkl

setup:
    rm -rf ./venv
    python -m venv venv
    just activate
    pip install -r requirements.txt
    mkdir -p pdfs
