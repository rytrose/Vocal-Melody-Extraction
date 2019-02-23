from flask import Flask, request
from werkzeug.utils import secure_filename
from werkzeug.serving import run_simple
import requests
import threading
import os.path as op
import time

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'wav'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/process", methods=['POST'])
def process_route():
    if 'file' not in request.files:
        return '', 400
    file = request.files['file']
    if file.filename == '':
        return '', 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(op.join("audio_files", filename))
        threading.Thread(target=extract_audio, args=(filename,)).start()
        return '', 201


def extract_audio(filename):
    print("Processing %s..." % filename)
    shimi_url = "http://localhost:5001/receive"
    files = {'file': open("test.txt", "rb")}
    res = requests.post(shimi_url, files=files)
    print("Sent file.", res.status_code)


if __name__ == '__main__':
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True, use_evalex=True)
