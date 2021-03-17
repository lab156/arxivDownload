from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional,\
                      GRU, Dropout, GlobalAveragePooling1D, Conv1D
from tensorflow.keras.models import Sequential

def lstm_model_one_layer(cfg):
    lstm_model = Sequential([
        Embedding(cfg['tot_words'], cfg['embed_dim'], 
                  input_length=cfg['max_seq_len'],
                  weights=[embed_matrix],
                  trainable=False),
        Bidirectional(LSTM(cfg['lstm_cells'], return_sequences=True)),
        GlobalAveragePooling1D(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstm_model.summary(print_fn=logger.info) 
    return lstm_model

def conv_model_globavgpool(cfg, logger):
    conv_m = Sequential([
        Embedding(cfg['tot_words'], cfg['embed_dim'],
                  input_length=cfg['max_seq_len'], #weights=[embed_matrix],
                  trainable=False),
        Conv1D(cfg['conv_filters'], cfg['kernel_size'], activation='relu'),
        GlobalAveragePooling1D(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    conv_m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    conv_m.summary(print_fn=logger.info) 
    return conv_m
