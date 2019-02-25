from flask import Flask, request
from werkzeug.utils import secure_filename
from werkzeug.serving import run_simple
from run_melodia import run_melodia
import VocalMelodyExtraction
from project.utils import load_model
from subprocess import Popen
import requests
import threading
import os.path as op
import os
import time

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'wav'}
MODEL_PATH = "pretrained_models/Seg"
OUTPUTS_PATH = "/media/ashis/data/rytrose/melody_extraction"


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


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
    name = "_".join(filename.split('/')[-1].split('.')[:-1])

    mp3_path = filename
    wav_path = ".".join(filename.split('.')[:-1] + ['wav'])

    if not op.exists(wav_path):
        command_string = "ffmpeg -i %s %s" % (mp3_path, wav_path)
        print("Running '%s'" % command_string)
        conversion = Popen(command_string.split(" "))
        conversion.wait()
        os.remove(mp3_path)

    # Run melodia
    run_melodia(wav_path, op.join(OUTPUTS_PATH, "melodia_outputs", "melodia_" + name + ".p"))

    # Run deep learning
    args = {
        "model_path": MODEL_PATH,
        "input_file": wav_path,
        "batch_size_test": 40,
        "output_file": op.join(OUTPUTS_PATH, "cnn_outputs", "cnn_" + name),
        "jetson": False
    }
    args_struct = Struct(**args)
    VocalMelodyExtraction.testing(args_struct)

    # shimi_cnn_url = "http://localhost:5001/receive_cnn"
    # files = {'file': open(op.join(OUTPUTS_PATH, "cnn_outputs", "cnn_" + name + ".txt"), "rb")}
    # res = requests.post(shimi_cnn_url, files=files)
    # print("Sent cnn file.", res.status_code)
    #
    # shimi_melodia_url = "http://localhost:5001/receive_melodia"
    # files = {'file': open(op.join(OUTPUTS_PATH, "melodia_outputs", "melodia_" + name + ".p"), "rb")}
    # res = requests.post(shimi_melodia_url, files=files)
    # print("Sent melodia file.", res.status_code)


if __name__ == '__main__':
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True, use_evalex=True)
