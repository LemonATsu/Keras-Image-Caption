import os, sys
import _pickle as cPickle
import urllib.request

import pandas as pd
import scipy.misc
import numpy as np

from keras.models import Model, load_model
from train import image_attention_model as iam
from train import image_experimental_model as iem
from train import image_inception_model as iim
from train import image_caption_model as icm
from train import image_caption_model_v2 as icm_v2
from train import image_vocab_history_model as ivhm
from train import image_vocab_history_model_v2 as ivhm_v2
from train import image_vocab_history_model_v3 as ivhm_v3
from train import image_vocab_history_model_v4 as ivhm_v4
from train import image_vocab_history_model_v5 as ivhm_v5
from beamsearch import beamsearch as bs
from beamsearch import unroll


def generate_caption(model, enc_map, dec_map, img, with_seq=False, max_len=10):
    gen = []
    st, ed = enc_map['<ST>'], enc_map['<ED>']
    cur = st

    seq = 0
    while len(gen) < max_len:
        if with_seq:
            X = [np.array([img]), np.array([cur]), np.array([seq])]
            seq += 1
        else:
            X = [np.array([img]), np.array([cur])]
        cur = np.argmax(model.predict(X)[0])
        if cur != ed:
            gen.append(dec_map[cur])
        else:
            break
    return ' '.join(gen)

def generate_k_best(model, enc_map, dec_map, tag_map, img, with_seq=False, with_vhist=False, k=4, max_len=10):
    ans = bs(model, enc_map, dec_map, tag_map, img, with_seq, with_vhist, k, max_len)
    gen = []
    for x in ans:
        if x == 1 : break
        if x != 0 : gen.append(dec_map[x])
    return  ' '.join(gen)

def generate_caption_test(model, enc_map, dec_map, img, max_len=16):
    st, ed = enc_map['<ST>'], enc_map['<ED>']
    img_mat = np.zeros((max_len, 256))
    img_mat[0,:] = img
    gen = []
    curr_cap = np.zeros((1, max_len))
    for i in range(0, max_len-1):
        cur = np.argmax(model.predict([np.array([img_mat]), np.array([curr_cap])])[0,i,:])
        if cur != ed:
            gen.append(dec_map[cur])
            curr_cap[0,i] = cur
        else:
            break
    return ' '.join(gen)

def eval_human(model, img_map, df_cap, enc_map, dec_map, tag_map, with_seq=False, with_vhist=False, k=4, size=1, max_len=10):
    for idx in np.random.randint(df_cap.shape[0], size=size):
        row = df_cap.iloc[idx]
        cap = eval(row['caption'])
        img_id = row['img_id']
        img = img_map[img_id]
        #gen = generate_caption(model, enc_map, dec_map, img)
        gen = generate_k_best(model, enc_map, dec_map, tag_map, img, k=k, with_seq=with_seq, with_vhist=with_vhist, max_len=max_len)
        print('[{}]'.format(img_id))
        print('[generated] {}'.format(gen))
        print('[groundtruth] {}'.format(' '.join([dec_map[cap[i]] for i in range(1,len(cap)-1)])))

if __name__ == '__main__':
    path = sys.argv[1]
    #model = load_model(path)
    dec_map = cPickle.load(open('dataset/text/dec_map.pkl', 'rb'))
    enc_map = cPickle.load(open('dataset/text/enc_map.pkl', 'rb'))
    tag_map = cPickle.load(open('dataset/tag_of_word_idx.pkl', 'rb'))

    img_train = cPickle.load(open('dataset/train_img2048.pkl', 'rb'))
    img_test = cPickle.load(open('dataset/test_img2048.pkl', 'rb'))
    df_cap = pd.read_csv('dataset/text/train_enc_cap.csv')

    model = ivhm_v5(clipnorm=1.)
    model.load_weights(path)

    eval_human(model, img_train, df_cap, enc_map, dec_map, tag_map, with_seq=True, with_vhist=True, k=1, size=40, max_len=13)

