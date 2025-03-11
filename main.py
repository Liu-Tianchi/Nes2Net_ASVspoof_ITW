import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils_SSL import genSpoof_list, Dataset_ASVspoof2019_train, Dataset_ASVspoof2021_eval, Dataset_ASVspoof2021_eval_no_cut
from tensorboardX import SummaryWriter
from startup_config import set_random_seed
import time

def pad(x, max_len):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x
def average_model(model, n_average_model, model_ID_to_average, best_save_path, model_folder_path):
    sd = None
    for ID in model_ID_to_average:
        model.load_state_dict(torch.load(os.path.join(model_folder_path, 'epoch_{}.pth'.format(ID))))
        print('Model loaded : {}'.format(os.path.join(model_folder_path, 'epoch_{}.pth'.format(ID))))
        if sd is None:
            sd = model.state_dict()
        else:
            sd2 = model.state_dict()
            for key in sd:
                sd[key] = (sd[key] + sd2[key])
    for key in sd:
        sd[key] = (sd[key]) / n_average_model
    model.load_state_dict(sd)
    torch.save(model.state_dict(), best_save_path)
    print('Model loaded average of {} best models in {}'.format(n_average_model, best_save_path))
def produce_evaluation_file(dataset, model, device, save_path, batch_size):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    for batch_x, utt_id in data_loader:
        fname_list = []
        score_list = []
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1]
        ).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()
    print('Scores saved to {}'.format(save_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument("--in_the_wild_path", type=str, default='/home/tianchi/SSL_Anti-spoofing/database/release_in_the_wild', help="the path of the folder of in the wild")
    parser.add_argument('--database_path', type=str, default='/home/tianchi/SSL_Anti-spoofing/database/LA/',
                        help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %   |- LA
    %      |- ASVspoof2021_LA_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac
    '''
    # model
    parser.add_argument("--dilation", type=int, default=2)
    # parser.add_argument("--num_stack", type=int, default=1, help="Number of Nes2Net blocks to stacked")
    parser.add_argument('--n_output_logits', type=int, default=2, 
                        help='number of output logits for the model, default is 2, following wav2vec2-AASIST repo')
    parser.add_argument('--date', type=str, default='unknown',
                        help='date')
    parser.add_argument('--model_name', type=str, required=True, choices=['wav2vec2_AASIST', 'wav2vec2_Nes2Net_X'],
                        help='the type of the model, check from the choices')
    parser.add_argument('--protocols_path', type=str, default='database/',
                        help='Change with path to user\'s LA database protocols directory address')
    parser.add_argument('--test_length', type=str, default='4s', choices=['full', '4s'],
                        help='length for testing. choose from 4s or full utterance')
    # parser.add_argument("--Nes_N", type=int, default=1)
    parser.add_argument("--pool_func", type=str, default='mean', choices=['mean', 'ASTP'],
                        help="pooling function, choose from mean and ASTP")
    parser.add_argument("--Nes_ratio", type=int, nargs='+', default=[8, 8], help="Nes_ratio, from outer to inner")
    # parser.add_argument("--scale", type=int, default=4, help="scale of inner")
    parser.add_argument("--test_protocol", type=str, default='4sec', choices=['4sec', 'full'], help="scale of inner")
    parser.add_argument("--SE_ratio", type=int, nargs='+', default=[1],
                        help="SE downsampling ratio in the bottleneck")
    parser.add_argument("--BTN_ratio", type=float, default=0.8, help="bottlenect ratio")
    '''
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
    '''
    # Hyperparameters
    parser.add_argument('--num_average_model', type=int, default=1)
    parser.add_argument("--model_ID_to_average", type=int, nargs='+', default=[], help="Nes_ratio, from outer to inner")
    parser.add_argument("--model_folder_path", type=str, help="the path of the folder of saved checkpoints")

    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed (default: 1234)')
    # model name and path
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='LA', choices=['LA', 'DF', 'in_the_wild'], help='LA/DF/in_the_wild')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False, help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True,
                        help='use cudnn-deterministic? (default true)')

    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False,
                        help='use cudnn-benchmark? (default false)')

    ##===================================================Rawboost data augmentation ======================================================================#
    parser.add_argument('--algo', type=int, default=8,
                        help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2), 9: parallel algo(1,2,3), 10: parallel algo(1,2) with possibility \
                          11: series algo (1+2+3) with possibility [default=0]')
    parser.add_argument('--LnL_ratio', type=float, default=1.0,
                    help='This is the possibility to activate LnL, which will only be used when algo>=10.')
    parser.add_argument('--ISD_ratio', type=float, default=1.0,
                    help='This is the possibility to activate ISD, which will only be used when algo>=10.')
    parser.add_argument('--SSI_ratio', type=float, default=1.0,
                    help='This is the possibility to activate SSI, which will only be used when algo>=11.')
    # LnL_convolutive_noise parameters
    parser.add_argument('--nBands', type=int, default=5,
                        help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20,
                        help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000,
                        help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100,
                        help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000,
                        help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10,
                        help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100,
                        help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0,
                        help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0,
                        help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5,
                        help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20,
                        help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5,
                        help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10,
                        help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2,
                        help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10,
                        help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40,
                        help='Maximum SNR value for coloured additive noise.[defaul=40]')

    ##===================================================Rawboost data augmentation ======================================================================#
    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()

    # make experiment reproducible
    set_random_seed(args.seed, args)
    track = args.track

    # database
    prefix = 'ASVspoof_{}'.format(track)
    prefix_2021 = 'ASVspoof2021.{}'.format(track)

    # define model saving path
    if args.algo < 10:
        model_tag = '{}_eff_{}_e{}_bz{}_lr{}_algo{}_seed{}'.format(args.date, args.model_name, args.num_epochs, args.batch_size, args.lr, args.algo, args.seed)
    else:  # ratio based augmentation approach @ Tianchi
        model_tag = '{}_eff_{}_e{}_bz{}_lr{}_algo{}_LnL{}_ISD{}_SSI{}'.format(args.date, args.model_name, args.num_epochs, args.batch_size, args.lr, args.algo, int(args.LnL_ratio*10), int(args.ISD_ratio*10), int(args.SSI_ratio*10))
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    # set model save directory
    if not os.path.exists(model_save_path) and not args.eval:
        os.mkdir(model_save_path)

    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    if args.model_name == 'wav2vec2_AASIST':
        from model_scripts.wav2vec2_AASIST import Model
    elif args.model_name == 'wav2vec2_Nes2Net_X':
        from model_scripts.wav2vec2_Nes2Net_X import wav2vec2_Nes2Net_no_Res_w_allT as Model
    else:
        raise ValueError
    model = Model(args, device).to(device)

    num_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    num_params_SSL = sum([param.view(-1).size()[0] for param in model.ssl_model.parameters()])
    print('num_params    :', num_params)
    print('num_params_SSL:', num_params_SSL)

    # set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # load model or average checkpoints
    if args.num_average_model > 1:  # average checkpoints
        assert len(args.model_ID_to_average) == args.num_average_model, ('num_average_model is not equal to the number of model IDs provided in model_ID_to_average')
        model_path_to_test = args.model_folder_path + '/Averaged_Model_IDs'
        for item in args.model_ID_to_average:
            model_path_to_test += "_{}".format(item)
        model_path_to_test += ".pth"
        if os.path.exists(model_path_to_test):
            print(f"File '{model_path_to_test}' already exists. Model averaging operation skipped.")
        else:
            print(f"File '{model_path_to_test}' does not exist. Proceeding with the model averaging operation...")
            average_model(model=model, n_average_model=args.num_average_model, model_ID_to_average=args.model_ID_to_average,
                      best_save_path=model_path_to_test, model_folder_path=args.model_folder_path)
        model.load_state_dict(torch.load(model_path_to_test, map_location=device))
        print('Model loaded : {}'.format(model_path_to_test))
    elif args.model_path:
        model_path_to_test = args.model_path
        model.load_state_dict(torch.load(model_path_to_test, map_location=device))
        print('Model loaded : {}'.format(model_path_to_test))

    # evaluation
    if args.eval:
        if track == 'LA' or track == 'DF':
            s = time.strftime("%a, %d %b %Y %I:%M:%S", time.gmtime())
            print('training started at:', s)
            st_t = time.time()
            file_eval = genSpoof_list(dir_meta=os.path.join(
                args.protocols_path + '{}_cm_protocols/{}.cm.eval.trl.txt'.format(prefix, prefix_2021)), is_train=False,
                                      is_eval=True)
            print('no. of eval trials', len(file_eval))
            if args.test_protocol == '4sec':
                eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir=os.path.join(
                    args.database_path + 'ASVspoof2021_{}_eval/'.format(args.track)))
            elif args.test_protocol == 'full':
                eval_set = Dataset_ASVspoof2021_eval_no_cut(list_IDs=file_eval, base_dir=os.path.join(
                    args.database_path + 'ASVspoof2021_{}_eval/'.format(args.track)))
            produce_evaluation_file(eval_set, model, device, args.eval_output, args.batch_size)
            en_t = time.time()
            print('time cost:', (en_t - st_t)/3600, 'hrs')
            sys.exit(0)
        elif track == 'in_the_wild':
            from tqdm import tqdm
            import librosa
            s = time.strftime("%a, %d %b %Y %I:%M:%S", time.gmtime())
            print('training started at:', s)
            st_t = time.time()
            with torch.no_grad():
                for filename in tqdm(sorted(os.listdir(args.in_the_wild_path)), desc=f"Testing"):
                    if filename.lower().endswith(".wav"):
                        audio_path = os.path.join(args.in_the_wild_path, filename)
                        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
                        if args.test_protocol == '4sec':
                            audio = pad(audio, 64000)
                        elif args.test_protocol == 'full':
                            pass
                        x = torch.tensor(audio).unsqueeze(0).to(device)
                        pred = model(x)[:, 1]
                        file_basename = os.path.splitext(filename)[0]
                        with open(args.eval_output, "a") as f:
                            f.write(f"{file_basename} {pred.item()}\n")
            en_t = time.time()
            print('time cost:', (en_t - st_t)/3600, 'hrs')
            sys.exit(0)
        else:
            raise ValueError
    # define train dataloader
    d_label_trn, file_train = genSpoof_list(
        dir_meta=os.path.join(args.protocols_path + 'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'),
        is_train=True, is_eval=False)

    print('no. of training trials', len(file_train))

    train_set = Dataset_ASVspoof2019_train(args, list_IDs=file_train, labels=d_label_trn, base_dir=os.path.join(args.database_path + 'ASVspoof2019_LA_train/'), algo=args.algo)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=6, shuffle=True, drop_last=True)

    del train_set, d_label_trn

    # define validation dataloader

    d_label_dev, file_dev = genSpoof_list(
        dir_meta=os.path.join(args.protocols_path + 'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'),
        is_train=False, is_eval=False)

    print('no. of validation trials', len(file_dev))

    dev_set_w_aug = Dataset_ASVspoof2019_train(args, list_IDs=file_dev,
                                         labels=d_label_dev,
                                         base_dir=os.path.join(
                                             args.database_path + 'ASVspoof2019_LA_dev/'), algo=args.algo)
    dev_set_wo_aug = Dataset_ASVspoof2019_train(args, list_IDs=file_dev,
                                         labels=d_label_dev,
                                         base_dir=os.path.join(
                                             args.database_path + 'ASVspoof2019_LA_dev/'), algo=0)
    dev_loader_w_aug = DataLoader(dev_set_w_aug, batch_size=args.batch_size, num_workers=6, shuffle=False)
    dev_loader_wo_aug = DataLoader(dev_set_wo_aug, batch_size=args.batch_size, num_workers=6, shuffle=False)
    del dev_set_w_aug, dev_set_wo_aug, d_label_dev

    # Training and validation
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    loss_min_w_aug = 999
    loss_min_wo_aug = 999
    epoch_loss_w_aug = 0
    epoch_loss_wo_aug = 0
    for epoch in range(1, num_epochs+1):
        if epoch == 0:
            s = time.strftime("%a, %d %b %Y %I:%M:%S", time.gmtime())
            print('training started at:', s)
        st_t = time.time()
        running_loss = 0
        num_total = 0.0
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x)
            batch_loss = criterion(batch_out, batch_y)
            running_loss += (batch_loss.item() * batch_size)
            batch_loss.backward()
            optimizer.step()
        running_loss /= num_total
        optimizer.zero_grad()
        del batch_loss, batch_x, batch_y, batch_out
        torch.cuda.empty_cache()

        val_loss_w_aug = 0.0
        val_loss_wo_aug = 0.0
        num_total = 0.0
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in dev_loader_w_aug:
                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_x = batch_x.to(device)
                batch_y = batch_y.view(-1).type(torch.int64).to(device)
                batch_out = model(batch_x)
                batch_loss = criterion(batch_out, batch_y)
                val_loss_w_aug += (batch_loss.item() * batch_size)
            val_loss_w_aug /= num_total
            optimizer.zero_grad()
            del batch_loss, batch_x, batch_y, batch_out
            torch.cuda.empty_cache()

            for batch_x, batch_y in dev_loader_wo_aug:
                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_x = batch_x.to(device)
                batch_y = batch_y.view(-1).type(torch.int64).to(device)
                batch_out = model(batch_x)
                batch_loss = criterion(batch_out, batch_y)
                val_loss_wo_aug += (batch_loss.item() * batch_size)
            val_loss_wo_aug /= num_total
            optimizer.zero_grad()
            del batch_loss, batch_x, batch_y, batch_out
            torch.cuda.empty_cache()
        save_flag = False
        if val_loss_w_aug <= loss_min_w_aug:
            loss_min_w_aug = val_loss_w_aug
            epoch_loss_w_aug = epoch
            save_flag = True
        if val_loss_wo_aug <= loss_min_wo_aug:
            loss_min_wo_aug = val_loss_wo_aug
            epoch_loss_wo_aug = epoch
            save_flag = True
        writer.add_scalar('val_loss_w_aug', val_loss_w_aug, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        writer.add_scalar('val_loss_wo_aug', val_loss_wo_aug, epoch)
        print('\n{}: tr_loss {:.5f}, -val_loss_w_aug {:.5f}, -val_loss_wo_aug {:.5f}, Best loss w/ aug: E{}/{:5f}, Best loss wo aug: E{}/{:.5f}'.format(epoch,
                                                   running_loss, val_loss_w_aug, val_loss_wo_aug, epoch_loss_w_aug, loss_min_w_aug, epoch_loss_wo_aug, loss_min_wo_aug))
        en_t = time.time()
        print('time cost:', en_t - st_t)
        if save_flag:
            torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))