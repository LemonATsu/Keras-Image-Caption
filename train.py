import os, sys
import _pickle as cPickle
import urllib.request

import pandas as pd
import scipy.misc
import numpy as np

from keras.models import load_model
from utils import *
from model import image_caption_model
from joblib import Parallel, delayed
import time
from keras.utils.layer_utils import print_summary


def gen_batch_in_thread(img_map, df_cap, vocab_size, n_jobs=4,
        size_per_thread=32):
    imgs , curs, nxts, seqs, vhists = [], [], [], [], []
    returns = Parallel(n_jobs=4, backend='threading')(
                            delayed(generate_batch)
			    (img_train, df_cap, vocab_size, size=size_per_thread) for i in range(0, n_jobs))

    for triple in returns:
        imgs.extend(triple[0])
        curs.extend(triple[1])
        nxts.extend(triple[2])
        seqs.extend(triple[3])
        vhists.extend(triple[4])

    return np.array(imgs), np.array(curs).reshape((-1,1)), np.array(nxts), \
            np.array(seqs), np.array(vhists)

def generate_batch(img_map, df_cap, vocab_size, size=32):
    imgs, curs, nxts, seqs, vhists = [], [], [], [], []

    for idx in np.random.randint(df_cap.shape[0], size=size):
        row = df_cap.iloc[idx]
        cap = eval(row['caption'])

        if row['img_id'] not in img_map.keys():
            continue

        img = img_map[row['img_id']]
        vhist = np.zeros((len(cap)-1, vocab_size))

        for i in range(1, len(cap)):
            seq = np.zeros((max_caplen))
            nxt = np.zeros((vocab_size))
            nxt[cap[i]] = 1
            curs.append(cap[i-1])
            seq[i-1] = 1

            if i < len(cap)-1:
                vhist[i, :] = np.logical_or(vhist[i, :], vhist[i-1, :])
                vhist[i, cap[i-1]] = 1

            nxts.append(nxt)
            imgs.append(img)
            seqs.append(seq)

        vhists.extend(vhist)

    return imgs, curs, nxts, seqs, vhists

if __name__ == '__main__':

    # initialization
    hist_path = 'history/'
    mdl_path = 'weights/'

    # read pkl
    dec_map = cPickle.load(open('dataset/text/dec_map.pkl', 'rb'))
    enc_map = cPickle.load(open('dataset/text/enc_map.pkl', 'rb'))

    img_train = cPickle.load(open('dataset/train_img2048.pkl', 'rb'))
    img_test = cPickle.load(open('dataset/test_img256.pkl', 'rb'))

    df_cap = pd.read_csv('dataset/text/train_enc_cap.csv')

    vocab_size = len(dec_map)
    embedding_matrix = generate_embedding_matrix('pre_trained/glove.6B.100d.txt', dec_map)
    model = image_caption_model(vocab_size=vocab_size, embedding_matrix=embedding_matrix)

    if len(sys.argv) >= 2:
        print('load weights from : {}'.format(sys.argv[1]))
        model.load_weights(sys.argv[1])

    # insert ur version name here
    version = 'v1.0.0'
    batch_num = 70
    print_summary(model.layers)

    hist_loss = []

    for i in range(0, 100):
        for j in range(0, batch_num):
            s = time.time()
	    # 64 x 128 = 8192 images per batch.
	    # 8 x 32 = 256 images for validation.
            img1, cur1, nxt1, seq1, vhists1 = gen_batch_in_thread(img_train, df_cap,
                                    vocab_size, n_jobs=64, size_per_thread=128)
            img2, cur2, nxt2, seq2, vhists2 = gen_batch_in_thread(img_train, df_cap, 
                                    vocab_size, n_jobs=8, size_per_thread=32)
            hist = model.fit([img1, cur1, seq1, vhists1], nxt1, batch_size=8192, nb_epoch=1, verbose=0,
                                    validation_data=([img2, cur2, seq2, vhists2], nxt2), shuffle=True)

            print("epoch {0}, batch {1} - training loss : {2}, validation loss: {3}"
                    .format(i, j, hist.history['loss'][-1], hist.history['val_loss'][-1]))
	    # record the training history
            hist_loss.extend(hist.history['loss'])

            if j % int(batch_num / 2) == 0 :
                print('check point')
                m_name = "{0}{1}_{2}_{3}_{4}.h5".format(mdl_path, version, i, j, time.time())
                model.save_weights(m_name)
                cPickle.dump({'loss':hist_loss}, open(hist_path+ 'history.pkl', 'wb'))

