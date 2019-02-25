import vamp
import soundfile as sf
import argparse
import numpy as np
import pickle


def run_melodia(input_path, output_path):
    audio, sr = sf.read(input_path, always_2d=True)
    audio = audio[:, 0]
    data = vamp.collect(audio, sr, "mtg-melodia:melodia")
    hop, frequencies = data['vector']

    timestamps = 8 * 128 / 44100.0 + np.arange(len(frequencies)) * (128 / 44100.0)

    pickle.dump({
        "frequencies": frequencies,
        "timestamps": timestamps
    }, open(output_path, "wb+"))

    print("FINISHED")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str)
    parser.add_argument('-o', '--output_path', type=str)
    args = parser.parse_args()

    run_melodia(args.input_path, args.output_path)
