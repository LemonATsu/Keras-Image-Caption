import os, sys
import _pickle as cPickle
import urllib.request

import pandas as pd
import scipy.misc
import numpy as np


from keras.models import Model, load_model
from model import image_caption_model
from beamsearch import beamsearch as bs
from beamsearch import unroll

def generate_k_best(model, enc_map, dec_map, img, k=4, max_len=10):
    ans = bs(model, enc_map, dec_map, tag_map, img, k, max_len)
    gen = []
    for x in ans:
        if x == 1 : break
        if x != 0 : gen.append(dec_map[x])
    return  ' '.join(gen)

def eval_human(model, img_map, df_cap, enc_map, dec_map, k=4, size=1, max_len=10):
    for idx in np.random.randint(df_cap.shape[0], size=size):
        row = df_cap.iloc[idx]
        cap = eval(row['caption'])
        img_id = row['img_id']
        img = img_map[img_id]
        gen = generate_k_best(model, enc_map, dec_map, img, k=k, max_len=max_len)
        print('[{}]'.format(img_id))
        print('[generated] {}'.format(gen))
        print('[groundtruth] {}'.format(' '.join([dec_map[cap[i]] for i in range(1,len(cap)-1)])))

if __name__ == '__main__':
    path = sys.argv[1]
    dec_map = cPickle.load(open('dataset/text/dec_map.pkl', 'rb'))
    enc_map = cPickle.load(open('dataset/text/enc_map.pkl', 'rb'))

    img_train = cPickle.load(open('dataset/train_img2048.pkl', 'rb'))
    img_test = cPickle.load(open('dataset/test_img2048.pkl', 'rb'))
    df_cap = pd.read_csv('dataset/text/train_enc_cap.csv')

    model = image_caption_model(clipnorm=1.)
    model.load_weights(path)

    eval_human(model, img_train, df_cap, enc_map, dec_map, with_seq=True, with_vhist=True, k=1, size=40, max_len=13)

