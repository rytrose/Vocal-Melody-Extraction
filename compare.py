import numpy as np
import matplotlib.pyplot as plt
import pretty_midi as pm

if __name__ == '__main__':
    midi = pm.PrettyMIDI('casey_jones.mid')
    data = np.loadtxt("out_seg.txt")
    data = data[:, 1]
    np.place(data, data == 0, np.nan)

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

    data_interp = np.interp(time, np.linspace(0, length, data.shape[0]), data)

    plt.plot(time, data_interp)
    plt.plot(time, midi_freqs)
    plt.legend(['Interpolated to Seconds DL', 'MIDI'])
    plt.show()

    plt.plot(np.linspace(0, 1, data.shape[0]), data)
    plt.plot(np.linspace(0, 1, len(time)), midi_freqs)
    plt.legend(['DL', 'MIDI'])
    plt.show()
