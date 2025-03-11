import argparse
import torch
import librosa
import numpy as np

def pad(x, max_len):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def main(args):
    # path = args.base_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create the model
    if args.model_name == 'wav2vec2_AASIST':
        from model_scripts.wav2vec2_AASIST import Model
    elif args.model_name == 'wav2vec2_Nes2Net_X':
        from model_scripts.wav2vec2_Nes2Net_X import wav2vec2_Nes2Net_no_Res_w_allT as Model
    else:
        raise ValueError
    model = Model(args, device).to(device)

    # Load the state dict of the saved model
    model_path_to_test = args.model_path
    model.load_state_dict(torch.load(model_path_to_test, map_location=device))
    print('Model loaded : {}'.format(model_path_to_test))

    model.eval()

    with torch.no_grad():
        audio, _ = librosa.load(args.file_to_test, sr=16000, mono=True)
        if args.test_mode == '4s':
            audio = pad(audio, 64000)
        x = torch.tensor(audio).unsqueeze(0).to(device)
        pred = model(x)[:, 1]
        print('score:', pred.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--n_output_logits", type=int, default=2)
    parser.add_argument('--model_name', type=str, required=True, choices=['wav2vec2_AASIST', 'wav2vec2_Nes2Net_X'],
                        help='the type of the model, check from the choices')
    parser.add_argument("--dilation", type=int, default=2, help="dilation")
    parser.add_argument("--pool_func", type=str, default='mean', choices=['mean', 'ASTP'],
                        help="pooling function, choose from mean and ASTP")
    parser.add_argument("--SE_ratio", type=int, nargs='+', default=[1], help="SE downsampling ratio in the bottleneck")
    parser.add_argument("--Nes_ratio", type=int, nargs='+', default=[8, 8], help="Nes_ratio, from outer to inner")
    parser.add_argument("--AASIST_scale", type=int, default=32, choices=[24, 32, 40, 48, 56, 64, 96],
                        help="the sacle of AASIST")
    # test config
    parser.add_argument("--file_to_test", type=str, required=True,
                        help="The path of the file to test.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="The path of the model checkpoint to test.")
    parser.add_argument("--test_mode", type=str, default='full', choices=['4s', 'full'],
                        help="Test using either the first 4 seconds or the full length. If the file is shorter than 4s, padding will be applied.")
    args = parser.parse_args()
    main(args)