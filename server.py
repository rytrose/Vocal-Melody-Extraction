from flask import Flask, request, send_from_directory
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
DATASET_PATH = "/media/ashis/data/rytrose/Lakh_MIDI"


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    e.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return op.join(msd_id[2], msd_id[3], msd_id[4], msd_id)


def msd_id_to_mp3(msd_id):
    """Given an MSD ID, return the path to the corresponding mp3"""
    return op.join(DATASET_PATH, 'lmd_matched_mp3', msd_id_to_dirs(msd_id) + '.mp3')


def msd_id_to_wav(msd_id):
    """Given an MSD ID, return the path to the corresponding mp3"""
    return op.join(DATASET_PATH, 'lmd_matched_mp3', msd_id_to_dirs(msd_id) + '.wav')


def msd_id_to_cnn(msd_id):
    """Given an MSD ID, return the path to its cnn processed melody extraction file"""
    return op.join(OUTPUTS_PATH, 'cnn_outputs', 'cnn_' + msd_id + '.txt')


def msd_id_to_melodia(msd_id):
    """Given an MSD ID, return the path to its cnn processed melody extraction file"""
    return op.join(OUTPUTS_PATH, 'melodia_outputs', 'melodia_' + msd_id + '.p')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/fetch/audio/<msdid>", methods=['GET'])
def fetch_audio(msdid):
    msd_id = msdid
    wav_path = msd_id_to_wav(msd_id)
    if not op.exists(wav_path):
        return 'No file exists at %s' % wav_path, 400

    split = wav_path.split("/")
    path = "/".join(split[:-1])
    filename = split[-1]

    return send_from_directory(path, filename)


@app.route("/fetch/cnn/<msdid>", methods=['GET'])
def fetch_cnn(msdid):
    msd_id = msdid
    cnn_path = msd_id_to_cnn(msd_id)
    if not op.exists(cnn_path):
        return 'No file exists at %s' % cnn_path, 400

    split = cnn_path.split("/")
    path = "/".join(split[:-1])
    filename = split[-1]

    return send_from_directory(path, filename)


@app.route("/fetch/melodia/<msdid>", methods=['GET'])
def fetch_melodia(msdid):
    msd_id = msdid
    melodia_path = msd_id_to_melodia(msd_id)
    if not op.exists(melodia_path):
        return 'No file exists at %s' % melodia_path, 400

    split = melodia_path.split("/")
    path = "/".join(split[:-1])
    filename = split[-1]

    return send_from_directory(path, filename)


@app.route("/process/<msdid>", methods=['GET'])
def process_msd_id(msdid):
    msd_id = msdid

    mp3_path = msd_id_to_mp3(msd_id)
    if not op.exists(mp3_path):
        return 'No msd_id %s found' % msd_id, 400

    threading.Thread(target=extract_audio, args=(msd_id,)).start()
    return 'Processing started.', 201


def extract_audio(msd_id):
    print("Processing %s..." % msd_id)
    name = msd_id

    mp3_path = msd_id_to_mp3(msd_id)
    wav_path = msd_id_to_wav(msd_id)

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

    r = requests.post("http://shimi-webapp-server.serveo.net/processed", json={'msdId': msd_id})


if __name__ == '__main__':
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True, use_evalex=True)
