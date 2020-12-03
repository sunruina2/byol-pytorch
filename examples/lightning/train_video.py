import argparse
import multiprocessing
from pathlib import Path
import time
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from byol_pytorch import BYOL
import pytorch_lightning as pl
import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
from torch.utils.data._utils.collate import default_collate  # 导入默认的拼接方式
import cv2
import lmdb
import pickle as pkl
from torchvision import transforms
from label_dct import labelname2cid_dct
import random

LR = 3e-4
print('####################################################################################', LR)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


class SetConfig():
    def __init__(self):
        self.test_flag = 1
        self.if_make_lmdb_vocab = 0
        self.pro_path = '/Users/sunruina/Documents/py_project/oversea_low_detection'
        if self.test_flag == 0:
            self.pro_path = '/home/sunruina/oversea_low_detection'
        self.task_name = 'BYOL_vfirstimg'
        self.run_start = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        self.gpu_N = torch.cuda.device_count()
        if self.if_make_lmdb_vocab == 1:
            self.info_v_tm = self.run_start
        else:
            self.info_v_tm = '2020_12_03_19_06_15'  # ind op add emb
        self.data_path = './train_data_process/'

        self.info_all_path = self.pro_path + '/train_op_tag/op_train_info/20201008/train_infos_20201008.csv'
        self.video_path_all = self.pro_path + '/train_op_tag/op_train_data/'

        self.model_dct_path = self.data_path + 'model_dct/'

        self.info_lmdb_trainpath = self.data_path + 'lm_db/' + self.info_v_tm + '_' + self.task_name + '_lmdb_train'
        self.info_lmdb_testpath = self.data_path + 'lm_db/' + self.info_v_tm + '_' + self.task_name + '_lmdb_test'
        self.eval_dt_lst = ['20201003', '20201004', '20201005', '20201006', '20201007', '20201008']

        self.vocab_word_path = self.data_path + 'vocab/' + self.info_v_tm + '_' + self.task_name + '_vocab_word' + '.vocab'
        self.vocab_word_max_size = 50000
        self.vocab_word_min_freq = 3

        self.vocab_char_path = self.data_path + 'vocab/' + self.info_v_tm + '_' + self.task_name + '_vocab_char' + '.vocab'
        self.vocab_char_max_size = 50000
        self.vocab_char_min_freq = 3

        self.cut_tms = 1
        self.skip_tms = 1
        # v_padding_size:  int(np.floor(self.cut_tms / self.skip_tms))

        self.label1_sig = 'op_l1_co'
        self.label2_sig = 'op_l2'
        self.target_type = 'l2'
        if self.target_type == 'l2':
            self.label_sig = self.label2_sig
        else:
            self.label_sig = self.label1_sig
        self.l2_w, self.l1_w = 1, 0
        self.use_bucket = ['ind']
        self.loss_type = 'focal'

        self.if_info_join = 0
        self.use_old = 1
        self.if_cnn_fineturn = 0
        self.if_rgb_cnn_fineturn = 1
        self.epoch_oldctn_N = 0
        self.epoch_save_startN = 0
        self.batch_size = 8 * self.gpu_N
        self.N_workers = multiprocessing.cpu_count()
        # need_set

        # training parameters
        # 按照cid排序
        name2cid_label_dct = labelname2cid_dct.get(self.label_sig)
        self.cid2name_label_dct = {}
        for k, v in name2cid_label_dct.items():
            self.cid2name_label_dct[v] = k
        self.class_n = len(self.cid2name_label_dct)  # number of target category

        self.class2_n = len(labelname2cid_dct.get(self.label2_sig))
        if self.class_n == 2:
            self.cid2name_label_dct[0] = '非' + self.cid2name_label_dct[1]
        self.epochs = 10  # training epochs
        self.learning_rate = 3e-4
        self.traineval_print = 10  # interval for displaying training info
        self.valid_interval = 100 * self.traineval_print  # interval for displaying valid info
        if self.test_flag == 1 or self.gpu_N == 0:
            self.use_old = 0
            self.traineval_print = 2  # interval for displaying training info
            self.valid_interval = 1 * self.traineval_print
            if self.gpu_N == 0:
                self.batch_size = 2  # 需要大于gpu数量*2，不然并行计算会报错
                self.eval_dt_lst = ['20200102']
            else:
                self.batch_size = self.gpu_N * 2  # 需要大于gpu数量*2，不然并行计算会报错
            self.info_lmdb_testpath = self.info_lmdb_trainpath
            self.epochs = 1

        # EncoderCNN architecture

        self.dropout_p = 0.3  # dropout probability

        # DecoderRNN architecture

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use CPU or GPU
        self.params = {'batch_size': self.batch_size, 'shuffle': True, 'num_workers': self.N_workers,
                       'pin_memory': True, 'drop_last': True}

        # set path
        self.cnn_flag = ""  # ['ResNetCRNN', 'ResNet_patch','ResCNNEncoder_diff_and_rgb']
        self.rnn_flag = ""  # ['bilstmatt_txt', 'base', 'reducesum', 'bilstmatt_video','decoderdiff', 'DecoderTrans', 'DecoderSum_txt_y12_emb', 'DecoderSum_txt_y12_emb_diff', 'Decoder_lstm_txt_y12_emb_diff']

        self.resnet_n, self.CNN_fc_hidden1, self.CNN_fc_hidden2, self.pici_embed_n = '50', 128, 128, 128
        self.num_layers, self.nhead, self.trans_hdim = 2, 4, 128

        self.patch_resnet_n = 18
        self.RNN_hidden_layers, self.RNN_hidden_nodes, self.RNN_FC_dim = 1, 128, 128
        self.model_name = self.run_start + "_" + self.task_name
        if not os.path.exists(self.model_dct_path):
            os.mkdir(self.model_dct_path)
        self.save_model_path = self.model_dct_path + self.model_name
        # self.trian_losspic_path = "/home/sunruina/crnn_video/saved_dct/" + self.model_name + '.jpg'
        self.if_build_vocab = 1
        if self.use_old == 1:
            self.learning_rate = 1e-5
            # self.old_cnn_path = './op_saved_dct/2020_10_13_12_53_04_ResNetCRNN_bilstmatt_txt_resnet50_checklmdb/cnn_encoder_epoch8_bn1.pth'
            # self.old_rnn_path = './op_saved_dct/2020_10_13_12_53_04_ResNetCRNN_bilstmatt_txt_resnet50_checklmdb/rnn_decoder_epoch8_bn1.pth'
            # self.old_cnn_path = self.model_dct_path + '2020_11_04_21_40_30_OpTag_ResNetCRNN_DecoderSum_txt/cnn_encoder_epoch1_bn11000.pth'
            # self.old_rnn_path = self.model_dct_path + '2020_11_04_21_40_30_OpTag_ResNetCRNN_DecoderSum_txt/rnn_dcoder_epoch1_bn11000.pth'
            self.old_cnn_path = self.model_dct_path + '2020_11_30_12_30_30_OpTagEmb_ResNetCRNN_DecoderSum_txt_y12_emb/cnn_encoder_epoch3_bn36.pth'
            self.old_rnn_path = self.model_dct_path + '2020_11_30_12_30_30_OpTagEmb_ResNetCRNN_DecoderSum_txt_y12_emb/rnn_decoder_epoch3_bn36.pth'

            # self.old_cnn_path = './op_saved_dct/2020_10_24_10_03_52_ResNetCRNN_bilstmatt_txt_resnet101_checklmdb/cnn_encoder_epoch1_bn162.pth'
            # self.old_rnn_path = '../op_saved_dct/2020_10_24_10_03_52_ResNetCRNN_bilstmatt_txt_resnet101_checklmdb/cnn_encoder_epoch1_bn162.pth'

        #
        # if self.use_old == 1:
        #     print('old_cnn_path', self.old_cnn_path)
        #     print('old_rnn_path', self.old_rnn_path)


cfg = SetConfig()


def prn_obj(obj_i):
    print('\n'.join(['%s:%s' % item for item in obj_i.__dict__.items()]))


prn_obj(cfg)

# test model, a resnet 50

resnet = models.resnet50(pretrained=True)


# arguments

# parser = argparse.ArgumentParser(description='byol-lightning-test')
#
# parser.add_argument('--image_folder', type=str, required=True,
#                     help='path to your folder of images for self-supervised learning')

# args = parser.parse_args()

# constants


# pytorch lightning module

class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        self.learner.update_moving_average()


# images dataset

def expand_greyscale(t):
    return t.expand(3, -1, -1)


def build_vocab(data_word, tokenizer, max_size, min_freq):
    # 词/字典
    vocab_dic = {}
    for content in data_word:  # 遍历每一行
        for word in tokenizer(content):  # 分词 or 分字
            vocab_dic[word] = vocab_dic.get(word, 0) + 1  # 构建词或字到频数的映射 统计词频/字频
    # 根据 min_freq过滤低频词，并按频数从大到小排序，然后取前max_size个单词
    print('build vacab all N :', len(vocab_dic))
    print('build vacab max_size N :', max_size)
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                 :max_size]
    # 构建词或字到索引的映射 从0开始
    print('build vacab min_freq :', min_freq)
    vocab_dic_select = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    print('build vacab min_freq N :', len(vocab_dic_select))
    # 添加未知符和填充符的映射
    vocab_dic_select.update({UNK: len(vocab_dic_select), PAD: len(vocab_dic_select) + 1})
    return vocab_dic_select, vocab_dic


def build_train_vocab(np_data, vocab_path, ues_word=0, vocab_max_size=50000, vocab_min_freq=5):
    # 构建词/字典
    print('\n\nbuild new vocab...if use_word:', ues_word)

    if ues_word == 1:  # 基于词 提前用分词工具把文本分开 以空格为间隔
        tokenizer = lambda x: x.split(' ')  # 直接以空格分开 word-level
    else:  # 基于字符
        tokenizer = lambda x: [y for y in x]  # char-level
    vocab_cntpath = vocab_path.split('.vocab')[0] + '.vocabcnt'
    vocab_select, vocab_allcnt = build_vocab(np_data, tokenizer, vocab_max_size,
                                             vocab_min_freq)

    # 保存构建好的词/字典
    pkl.dump(vocab_select, open(vocab_path, 'wb'))
    pkl.dump(vocab_allcnt, open(vocab_cntpath, 'wb'))
    print('build new vocab finish!!!')
    return vocab_select


def make_lmdb_vocab_op():
    print('make new lmdb...')
    train_env = lmdb.open(cfg.info_lmdb_trainpath, map_size=1099511627776)
    train_env_writing = train_env.begin(write=True)
    test_env = lmdb.open(cfg.info_lmdb_testpath, map_size=1099511627776)
    test_env_writing = test_env.begin(write=True)
    good_ntrain = 0
    good_ntest = 0
    badcase_n = 0
    txt_list = []
    with open(cfg.info_all_path, 'r', encoding='utf-8') as file:
        for ii, line in enumerate(file):
            if ii % 10000 == 0:
                print(ii, 'good_ntrain good_ntest bad_n:', good_ntrain, good_ntest, badcase_n)
            if 'photo_id' in line:
                continue
            # data_all = line[:-1].split('\x00')
            # s1, s2, s3 = data_all[0], data_all[1].replace(',', '_'), data_all[2]
            # data_all = ''.join([s1, s2, s3])
            data_all_split = line.split(',')
            if len(data_all_split) == 13:
                # # photo_id, author_id, explore, txt, createdt,tags,tag_l1_id, tag_l2_id, tag_l1_name, tag_l2_name, cover_url, url, embedding, p_date
                photo_id = data_all_split[0]
                # author_id = data_all_split[1]
                explore = data_all_split[2]
                txt = data_all_split[3]
                createdt = data_all_split[4]
                # # tags = data_all_split[5]
                # tag_l1_id = data_all_split[5]
                # tag_l2_id = data_all_split[6]
                tag_l1_name = data_all_split[7]
                if 'LGBTQ' in tag_l1_name or explore not in cfg.use_bucket:
                    continue
                # tag_l2_name = data_all_split[8]
                # cover_url = data_all_split[9]
                # url = data_all_split[10]
                embedding = data_all_split[11]
                # p_date = data_all_split[12]

                video_path = cfg.video_path_all + createdt + '/2400/video_folder/' + photo_id + '.mp4'
                if os.path.exists(video_path):
                    line = ','.join([line[0:-1], video_path, embedding])
                    if createdt not in cfg.eval_dt_lst:
                        train_env_writing.put(key=str(good_ntrain).encode(), value=line.encode())
                        good_ntrain += 1
                        if txt != '':
                            txt_list.append(txt)
                    else:
                        test_env_writing.put(key=str(good_ntest).encode(), value=line.encode())
                        good_ntest += 1
            else:
                badcase_n += 1
                print('lmdb bad case', badcase_n, data_all_split[0])
    train_env_writing.commit()
    train_env.close()
    test_env_writing.commit()
    test_env.close()
    print("finish making lmdb with 'good_ntrain good_ntest bad_n':", good_ntrain, good_ntest, badcase_n)

    build_train_vocab(np.asarray(txt_list), cfg.vocab_word_path, ues_word=1,
                      vocab_max_size=cfg.vocab_word_max_size, vocab_min_freq=cfg.vocab_word_min_freq)
    build_train_vocab(np.asarray(txt_list), cfg.vocab_char_path, ues_word=0,
                      vocab_max_size=cfg.vocab_char_max_size, vocab_min_freq=cfg.vocab_char_min_freq)


def cv2_video_random_reshape(v_path, if_random_start=0, cut_tms=5, skip_tms=0.1, edge_maxsize=None, frames_path=None,
                             if_pad_img=0):
    """

    Args:
        v_path: a video path or url or 'rb'file: string
        cut_tms: length of clipping time sunch as 10 seconds: int
        skip_tms: how many seconds to skip for saving a frame: float

    Returns: frames array : list

    """
    if not os.path.exists(v_path):
        print('v_path not exist:', v_path)
        return None, None
    cap = cv2.VideoCapture(v_path)
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 25
    cut_frames_n = cut_tms * fps
    if n_frame > cut_frames_n:
        gap_n = n_frame - cut_frames_n
        if if_random_start == 0:
            start_frame = 0
        else:
            start_frame = random.randint(0, gap_n)
        end_frame = start_frame + cut_frames_n
    else:

        start_frame = 0
        end_frame = min(cut_frames_n, n_frame)
    seq = []
    skip_fps = max(int(fps * skip_tms), 1)
    for i in range(end_frame):  # int(fps)):
        if i % skip_fps != 0 or i < start_frame:
            is_succeed = cap.grab()
        else:
            ret, img = cap.read()
            if img is not None:
                seq.append(img)
    h, w = seq[0].shape[0], seq[0].shape[1]
    origin_size = h * w

    if edge_maxsize is not None:
        pad_short = int(abs(w - h) / 2)
        if if_pad_img == 1:
            if h < w:
                pad_lurb_lst = (0, pad_short, 0, pad_short)
            else:
                pad_lurb_lst = (pad_short, 0, pad_short, 0)

            trans = transforms.Compose([
                transforms.Pad(pad_lurb_lst),
                transforms.Resize([edge_maxsize, edge_maxsize]),
            ])
        else:
            trans = transforms.Compose([
                transforms.Resize([edge_maxsize, edge_maxsize]),
                # transforms.Resize(edge_maxsize),
                # transforms.CenterCrop(edge_maxsize),
            ])
        seq = [trans(Image.fromarray(i[..., ::-1])) for i in seq]
    if frames_path is not None:
        if not os.path.exists(frames_path):
            os.mkdir(frames_path)
        for i, f_i in enumerate(seq):
            f_i.save(frames_path + "/frame" + str(i).zfill(6) + '.jpg')
    return seq, origin_size


class Dataset_CRNN_lmdb_op(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, lmdb_path, vocab_w, vocab_c, label_sig, train_flag=1, cut_tms=2, skip_tms=0.5, label2_sig=None):
        "Initialization"
        self.label_sig = label_sig
        self.label2_sig = label2_sig
        self.labelname2cid_dct = labelname2cid_dct

        env = lmdb.open(lmdb_path, readonly=True)
        self.env_writing = env.begin()
        self.lmdb_len = 0
        for key, value in self.env_writing.cursor():
            self.lmdb_len += 1

        self.vocab_w = vocab_w
        self.vocab_c = vocab_c
        if os.path.exists(self.vocab_w):
            self.vocab_w = pkl.load(open(self.vocab_w, 'rb'))
        if os.path.exists(self.vocab_c):
            self.vocab_c = pkl.load(open(self.vocab_c, 'rb'))
        self.cut_tms = cut_tms
        self.skip_tms = skip_tms

        self.if_pad_img = 0
        self.if_save_frames = 0
        self.v_pading_size, self.w_pading_size, self.c_pading_size = int(
            np.floor(self.cut_tms / self.skip_tms)), 32, 128
        self.v_img_size = 224
        if train_flag == 1:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.if_random_start = 1
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.if_random_start = 0

    def __len__(self):
        "Denotes the total number of samples"
        return self.lmdb_len

    def etl_frames_v2(self, path_v_i):
        try:

            '''V2'''

            frames, pixel_n = cv2_video_random_reshape(path_v_i, if_random_start=self.if_random_start,
                                                       cut_tms=self.cut_tms, skip_tms=self.skip_tms,
                                                       edge_maxsize=self.v_img_size,
                                                       frames_path=None, if_pad_img=self.if_pad_img)
            len_frames_i = len(frames)
            if len_frames_i >= self.v_pading_size:
                frames = frames[0:self.v_pading_size]
            else:
                pass
                # w, h = frames[0].size[0], frames[0].size[1]
                for i in range(self.v_pading_size - len_frames_i):
                    # frames.append(Image.fromarray(np.uint8(np.zeros((w, h, 3)))))
                    frames.append(Image.fromarray(np.uint8(np.zeros((self.v_img_size, self.v_img_size, 3)))))
            return frames

        except Exception as e:
            print('read_video fails id:', path_v_i)
            print('read_video fails  error:', e)
            return None

    def trans_vocab(self, content, vocab_info, pad_size, ues_word=True):
        '''sentence process'''
        # content_cut = list(jieba.cut(content, cut_all=True))
        words_line = []
        # 定义tokenizer函数（word-level/character-level）
        if ues_word:  # 基于词 提前用分词工具把文本分开 以空格为间隔
            tokenizer = lambda x: x.split(' ')  # 直接以空格分开 word-level
            token = tokenizer(content)  # 对文本进行分词/分字
        else:  # 基于字符
            tokenizer = lambda x: [y for y in x]  # char-level
            token = tokenizer(content)  # 对文本进行分词/分字
        if pad_size:  # 长截短填
            if len(token) < pad_size:  # 文本真实长度比填充长度 短
                token.extend([vocab_info.get(PAD)] * (pad_size - len(token)))  # 填充
            else:  # 文本真实长度比填充长度 长
                token = token[:pad_size]  # 截断
        # word to id
        for word in token:  # 将词/字转换为索引，不在词/字典中的 用UNK对应的索引代替
            words_line.append(vocab_info.get(word, vocab_info.get(UNK)))

        return torch.tensor(words_line)

    def __getitem__(self, index):
        value_string_read = self.env_writing.get(str(index).encode())
        if value_string_read is not None:
            data_all_split = value_string_read.decode().split(',')

            # # photo_id = data_all_split[0]
            # # author_id = data_all_split[1]
            # # explore = data_all_split[2]
            # txt = data_all_split[3]
            # # createdt = data_all_split[4]
            # tag_l1_id = data_all_split[5]
            # tag_l2_id = data_all_split[6]
            # tag_l1_name = data_all_split[7].replace(' ', '').replace('_', '')
            # tag_l2_name = data_all_split[8].replace(' ', '').replace('_', '')
            # # cover_url = data_all_split[9]
            # # url = data_all_split[10]
            # # p_date = data_all_split[12]
            video_path = data_all_split[13]
            # embedding = [float(ii) for ii in data_all_split[14].split('；')]
            #
            # label_cid, label2_cid = 0, 0
            # if 'op_l1' in self.label_sig:
            #     l1_idname = '_'.join([tag_l1_id, tag_l1_name])
            #     label_cid = self.labelname2cid_dct.get(self.label_sig).get(l1_idname, 0)
            #     if self.label2_sig is not None:
            #         l2_idname = '_'.join([tag_l1_id, tag_l1_name, tag_l2_id, tag_l2_name])
            #         label2_cid = self.labelname2cid_dct.get(self.label2_sig).get(l2_idname, 0)
            # else:
            #     l12_idname = '_'.join([tag_l1_id, tag_l1_name, tag_l2_id, tag_l2_name])
            #     label_cid = self.labelname2cid_dct.get(self.label_sig).get(l12_idname, 0)
            X_frames = self.etl_frames_v2(video_path)
            if X_frames is not None:
                X_frames = torch.stack([self.transform(frame_i) for frame_i in X_frames], dim=0)
                # X_wtxt = self.trans_vocab(txt, vocab_info=self.vocab_w, pad_size=self.w_pading_size, ues_word=True)
                # X_ctxt = self.trans_vocab(txt, vocab_info=self.vocab_c, pad_size=self.c_pading_size, ues_word=False)
                # y = torch.LongTensor([label_cid])  # (labels) LongTensor are for int64 instead of FloatTensor
                # y2 = torch.LongTensor([label2_cid])  # (labels) LongTensor are for int64 instead of FloatTensor
                # ytb_emb = torch.tensor(embedding, dtype=torch.float64)
                if self.v_pading_size == 1:
                    X_frames = X_frames.squeeze(dim=0)
                return X_frames
                # return X_frames, X_wtxt, X_ctxt, ytb_emb, y, y2
                # return None
            else:
                return None
        else:
            return None


def my_collate_fn(batch):
    '''
    batch中每个元素形如(data, label)
    '''
    # 过滤为None的数据
    batch_new = list(filter(lambda x: x is not None, batch))
    if len(batch_new) == 0:
        print('\nDataloader bug: current batch have not valid samples @my_collate_fn!')
    return default_collate(batch_new)  # 用默认方式拼接过滤后的batch数据


if __name__ == '__main__':
    # nohup python -u ResNetCRNN/ss4_train_ResNetCRNN_txt.py > train_log/train_bilstmatt_txt.log &
    LR = cfg.learning_rate
    if cfg.if_make_lmdb_vocab == 1:
        make_lmdb_vocab_op()

    '''data loader'''
    train_iter, valid_iter = Dataset_CRNN_lmdb_op(cfg.info_lmdb_trainpath, cfg.vocab_word_path, cfg.vocab_char_path,
                                                  cfg.label_sig, train_flag=1, cut_tms=cfg.cut_tms,
                                                  skip_tms=cfg.skip_tms, label2_sig=cfg.label2_sig), \
                             Dataset_CRNN_lmdb_op(cfg.info_lmdb_testpath, cfg.vocab_word_path, cfg.vocab_char_path,
                                                  cfg.label_sig, train_flag=0, cut_tms=cfg.cut_tms,
                                                  skip_tms=cfg.skip_tms, label2_sig=cfg.label2_sig)
    prn_obj(train_iter)
    train_loader = data.DataLoader(train_iter, **cfg.params, collate_fn=my_collate_fn)
    valid_loader = data.DataLoader(valid_iter, **cfg.params, collate_fn=my_collate_fn)

    model = SelfSupervisedLearner(
        resnet,
        image_size=224,
        hidden_layer='avgpool',
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99
    )

    trainer = pl.Trainer(
        gpus=cfg.gpu_N,
        max_epochs=cfg.epochs,
        accumulate_grad_batches=1
    )
    for i in range(1000):
        if i == 0:
            print('\n\nTraining:', i)
            trainer.fit(model, train_loader)
            if not os.path.exists(cfg.save_model_path):
                os.mkdir(cfg.save_model_path)
            save_path_i = cfg.save_model_path + "/epoch_" + str(i * cfg.epochs) + ".ckpt"
            trainer.save_checkpoint(save_path_i)
            print('model saved:', save_path_i)

        else:
            print('\n\nTraining:', i)
            trainer = pl.Trainer(
                resume_from_checkpoint=cfg.save_model_path + "/epoch_" + str((i - 1) * cfg.epochs) + ".ckpt")
            trainer.fit(model, train_loader)
            save_path_i = cfg.save_model_path + "/epoch_" + str(i * cfg.epochs) + ".ckpt"
            trainer.save_checkpoint(save_path_i)
            print('model saved:', save_path_i)
