from keras.applications.inception_v3 import InceptionV3
from model import image_caption_model
from keras.models import load_model
from keras.optimizers import RMSprop, Adadelta, SGD
from keras.utils.layer_utils import print_summary
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from utils import *
import scipy.misc
import numpy as np
import pandas as pd
import _pickle as cPickle
from evaluate import generate_k_best
from extractor import ImageFeatureExtractor

class ImageCaptionModel(object):

    def __init__(self, w_path, dec_path='dataset/text/dec_map.pkl',
                    enc_path='dataset/text/enc_map.pkl',
                    embedding_path='pre_trained/glove.6B.100d.txt'):

        dec_map = cPickle.load(open(dec_path, 'rb'))
        enc_map = cPickle.load(open(enc_path, 'rb'))

        embedding_matrix = generate_embedding_matrix(embedding_path, dec_map)
        self.model = image_caption_model(embedding_matrix=embedding_matrix)

        self.extractor = ImageFeatureExtractor('weights/tensorflow_inception_graph.pb')
        self.model.load_weights(w_path)
        self.dec_map = dec_map
        self.enc_map = enc_map

    def predict(self, img_path):
        img_feature = self.extractor.extract_features(img_path, flag_from_file=True)
        #print(img_feature)
        sentence = generate_k_best(self.model, self.enc_map, self.dec_map, img_feature, k=6, max_len=15)
        return sentence

if __name__ == '__main__':
    print('Initializing ...')

    # loading feature
    import sys
    weights = sys.argv[1]
    image_caption = ImageCaptionModel(weights)
    while True :
        n = input('Image Path (input quit to exit) :')
        if n == 'quit':
            break
        try:
            print('Resulting Caption ... : ')
            print(image_caption.predict(n))
            print('\n\n')
        except:
            print('cannot find file {}'.format(n))
    import gc; gc.collect()

