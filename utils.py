import numpy as np

def count_vocab_occurance(vocab, df):
    voc_cnt = {v:0 for v in vocab}
    for img_id, row in df.iterrows():
        for w in row['caption'].split(' '):
            voc_cnt[w] += 1
    return voc_cnt

def decode(dec_map, ids):
    return ' '.join([dec_map[x] for x in ids])

def generate_embedding_matrix(w2v_path, dec_map, lang_dim=100):
    out_vocab = []
    embeddings_index = {}
    f = open(w2v_path, 'r', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    # prepare embedding matrix
    embedding_matrix = np.random.rand(len(dec_map), lang_dim)
    for idx, wd in dec_map.items():
        if wd in embeddings_index.keys():
            embedding_matrix[idx] = embeddings_index[wd]
        else:
            out_vocab.append(wd)
    # print('words: "{}" not in pre-trained vocabulary list'.format(','.join(out_voca
    return embedding_matrix

