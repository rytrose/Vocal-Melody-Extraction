import argparse

import numpy as np
import soundfile as sf

from project.MelodyExt import feature_extraction
from project.utils import load_model, save_model, matrix_parser
from project.test import inference
from project.model import seg, seg_pnn, sparse_loss
from project.train import train_audio

# Only run 10s at a time
MAX_LEN = 44100 * 10

def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--phase',
                        help='phase: training or testing (default: %(default)s',
                        type=str, default='testing')

    #arguments for training
    parser.add_argument('-t', '--model_type',
                        help='model type: seg or pnn (default: %(default)s',
                        type=str, default='seg')
    parser.add_argument('-d', '--data_type',
                        help='data type: audio or symbolic (default: %(default)s',
                        type=str, default='audio')
    parser.add_argument('-da', '--dataset_path', nargs='+',
                        help='path to data set (default: %(default)s',
                        type=str, default='dataset')
    parser.add_argument('-la', '--label_path', nargs='+',
                        help='path to data set label (default: %(default)s',
                        type=str, default='dataset_label')
    parser.add_argument('-ms', '--model_path_symbolic',
                        help='path to symbolic model (default: %(default)s',
                        type=str, default='model_symbolic')

    parser.add_argument('-w', '--window_width',
                        help='width of the input feature (default: %(default)s',
                        type=int, default=128)
    parser.add_argument('-b', '--batch_size_train',
                        help='batch size during training (default: %(default)s',
                        type=int, default=12)
    parser.add_argument('-e', '--epoch',
                        help='number of epoch (default: %(default)s',
                        type=int, default=5)
    parser.add_argument('-n', '--steps',
                        help='number of step per epoch (default: %(default)s',
                        type=int, default=6000)

    parser.add_argument('-o', '--output_model_name',
                        help='name of the output model (default: %(default)s',
                        type=str, default="out")

    #arguments for testing
    parser.add_argument('-m', '--model_path',
                        help = 'path to existing model (default: %(default)s',
                        type = str, default = 'transfer_audio_directly')
    parser.add_argument('-i', '--input_file',
                        help='path to input file (default: %(default)s',
                        type=str, default='train01.wav')
    parser.add_argument('-bb', '--batch_size_test',
                        help='batch size during testing (default: %(default)s',
                        type=int, default=10)
    parser.add_argument('-of', '--output_file',
                        help='output file name (default: %(default)s',
                        type=str, default='out_seg')
    parser.add_argument('-j', '--jetson',
                        help='jetson flag (limit memory use)',
                        action='store_true', default=False)

    args = parser.parse_args()
    print(args)

    if(args.phase == "training"):
        #arguments setting
        TIMESTEPS = args.window_width

        #dataset_path = ["medleydb_48bin_all_4features", "mir1k_48bin_all_4features"]
        #label_path = ["medleydb_48bin_all_4features_label", "mir1k_48bin_all_4features_label"]
        dataset_path = args.dataset_path
        label_path = args.label_path


        # load or create model
        if("seg" in args.model_type):
            model = seg(multi_grid_layer_n=1, feature_num=384, input_channel=1, timesteps=TIMESTEPS)
        elif("pnn" in args.model_type):
            model = seg_pnn(multi_grid_layer_n=1, feature_num=384, timesteps=TIMESTEPS,
                            prev_model=args.model_path_symbolic)

        model.compile(optimizer="adam", loss={'prediction': sparse_loss}, metrics=['accuracy'])

        #train
        train_audio(model,
                    args.epoch,
                    args.steps,
                    args.batch_size_train,
                    args.window_width,
                    dataset_path,
                    label_path)

        #save model
        save_model(model, args.output_model_name)
    else:
        # load wav
        song = args.input_file
        x, fs = sf.read(song)
        results = None
        if args.jetson:
            sample_ptr = 0
            while sample_ptr < x.shape[0]:
                chunk_end = min(sample_ptr + MAX_LEN, x.shape[0] - 1)
                chunk = x[sample_ptr:chunk_end, :]
                sample_ptr += MAX_LEN

                # Feature extraction
                feature = feature_extraction(chunk, fs)
                feature = np.transpose(feature[0:4], axes=(2, 1, 0))

                # load model
                model = load_model(args.model_path)

                # Inference
                print(feature[:, :, 0].shape)
                extract_result = inference(feature= feature[:, :, 0],
                                           model = model,
                                           batch_size=args.batch_size_test)

                # Output
                r = matrix_parser(extract_result)

                if results is None:
                    results = r
                else:
                    results = np.concatenate((results, r))
        else:
            # Feature extraction
            feature = feature_extraction(x, fs)
            feature = np.transpose(feature[0:4], axes=(2, 1, 0))

            # load model
            model = load_model(args.model_path)

            # Inference
            print(feature[:, :, 0].shape)
            extract_result = inference(feature=feature[:, :, 0],
                                       model=model,
                                       batch_size=args.batch_size_test)

            # Output
            results = matrix_parser(extract_result)

        np.savetxt(args.output_file + ".txt", results)
        print("FINISHED")


if __name__ == '__main__':
    main()
