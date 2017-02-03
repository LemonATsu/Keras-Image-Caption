from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation
from keras.optimizers import RMSprop

def image_caption_model(vocab_size=2187, embedding_matrix=None, lang_dim=100,
            max_caplen=53, img_dim=2048, clipnorm=1):
    print('generating vocab_history model v5')
    # text: current word
    lang_input = Input(shape=(1,))
    img_input = Input(shape=(img_dim,))
    seq_input = Input(shape=(max_caplen,))
    vhist_input = Input(shape=(vocab_size,))

    if embedding_matrix is not None:
        x = Embedding(output_dim=lang_dim, input_dim=vocab_size, init='glorot_uniform', input_length=1, weights=[embedding_matrix])(lang_input)
    else:
        x = Embedding(output_dim=lang_dim, input_dim=vocab_size, init='glorot_uniform', input_length=1)(lang_input)

    lang_embed = Reshape((lang_dim,))(x)
    lang_embed = merge([lang_embed, seq_input], mode='concat', concat_axis=-1)
    lang_embed = Dense(lang_dim)(lang_embed)
    lang_embed = Dropout(0.25)(lang_embed)

    merge_layer = merge([img_input, lang_embed, vhist_input], mode='concat', concat_axis=-1)
    merge_layer = Reshape((1, lang_dim+img_dim+vocab_size))(merge_layer)

    gru_1 = GRU(img_dim)(merge_layer)
    gru_1 = Dropout(0.25)(gru_1)
    gru_1 = Dense(img_dim)(gru_1)
    gru_1 = BatchNormalization()(gru_1)
    gru_1 = Activation('softmax')(gru_1)

    attention_1 = merge([img_input, gru_1], mode='mul', concat_axis=-1)
    attention_1 = merge([attention_1, lang_embed, vhist_input], mode='concat', concat_axis=-1)
    attention_1 = Reshape((1, lang_dim + img_dim + vocab_size))(attention_1)
    gru_2 = GRU(1024)(attention_1)
    gru_2 = Dropout(0.25)(gru_2)
    gru_2 = Dense(vocab_size)(gru_2)
    gru_2 = BatchNormalization()(gru_2)
    out = Activation('softmax')(gru_2)

    model = Model(input=[img_input, lang_input, seq_input, vhist_input], output=out)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0001, clipnorm=1.))
    return model

