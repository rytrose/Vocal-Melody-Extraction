import numpy as np
import matplotlib.pyplot as plt
import pretty_midi as pm
import pickle

if __name__ == '__main__':
    midi = pm.PrettyMIDI('casey_jones.mid')

    melodia_data = pickle.load(open('melodia_out.p', 'rb'))
    _, melodia_data = melodia_data['vector']
    np.place(melodia_data, melodia_data <= 0, np.nan)
    melodia_timestamps = 8 * 128 / 44100.0 + np.arange(len(melodia_data)) * (128 / 44100.0)

    cnn_data = np.loadtxt("out_seg.txt")
    cnn_data = cnn_data[:, 1]


    start_stops = [[n.start, n.end] for n in midi.instruments[0].notes]

    length = start_stops[-1][-1]

    t = 0

    midi_freqs = []
    time = []

    while t < length:
        freq = np.nan
        for n in midi.instruments[0].notes:
            if n.start <= t <= n.end:
                freq = pm.note_number_to_hz(n.pitch)
                break
        midi_freqs.append(freq)
        time.append(t)
        t += 0.1

    midi_data_interp = np.interp(melodia_timestamps, time, midi_freqs)
    cnn_data_interp = np.interp(melodia_timestamps, np.linspace(0, length, cnn_data.shape[0]), cnn_data)
    np.place(cnn_data_interp, cnn_data_interp <= 4000, np.nan)

    plt.plot(melodia_timestamps, melodia_data)
    plt.plot(melodia_timestamps, cnn_data_interp)
    plt.plot(melodia_timestamps, midi_data_interp)
    plt.legend(['Melodia', 'CNN', 'MIDI'])
    plt.show()
