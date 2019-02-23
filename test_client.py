from flask import Flask, request
from werkzeug.utils import secure_filename
from werkzeug.serving import run_simple
import requests
import threading
import os.path as op
import time

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'txt', 'p'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/receive", methods=['POST'])
def receive_route():
    if 'file' not in request.files:
        return '', 400
    file = request.files['file']
    if file.filename == '':
        return '', 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(filename)
        return '', 201


def test():
    wait_len = 2
    for i in range(wait_len):
        print("%d..." % (wait_len - i))
        time.sleep(1)

    server_url = "http://localhost:5000/process"
    files = {'file': open("casey_jones.wav", "rb")}
    res = requests.post(server_url, files=files)
    print("Posted audio file.", res.status_code)


if __name__ == '__main__':
    run_simple('localhost', 5001, app, use_reloader=True, use_debugger=True, use_evalex=True)
