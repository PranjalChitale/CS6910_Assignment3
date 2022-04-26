#Reference : Tensorflow Documentation.
#https://www.tensorflow.org/guide/keras/rnn

import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import SimpleRNN,LSTM,GRU,Embedding,Dense,Input
from tensorflow.keras.optimizers import Adam,Nadam
#import wandb
#from wandb.keras import WandbCallback
import numpy as np


class Model(object):
    def __init__(self, english_char_set, indic_char_set, max_seq_len_english_encoder, max_seq_len_indic_decoder, indic_char_to_idx, indic_idx_to_char, cell ="LSTM", num_epochs =10, batch_size = 64, embedding_size = 32, num_enc_layers = 5, num_dec_layers =2, num_hidden_layers = 3, dropout = 0):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.len_enc_charset = len(english_char_set)
        self.len_dec_charset = len(indic_char_set)
        self.max_seq_len_english_encoder = max_seq_len_english_encoder
        self.max_decoder_seq_length = max_seq_len_indic_decoder
        self.indic_char_to_idx = indic_char_to_idx
        self.indic_idx_to_char = indic_idx_to_char
        self.cell = cell
        self.embedding_size = embedding_size
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers= num_dec_layers
        self.num_hidden_layers =num_hidden_layers
        self.encoder_model = None
        self.decoder_model = None
        self.model = None
        

    def build(self):
        encoder_inputs = Input(shape=(None,))
        encoder_context = Embedding(input_dim = self.len_enc_charset + 1, output_dim = self.embedding_size, input_length = self.max_decoder_seq_length )(encoder_inputs)
        encoder_outputs = encoder_context
        encoder_states = list()
        for j in range(self.num_enc_layers):
            if self.cell == "rnn":
                encoder_outputs, state = SimpleRNN(self.num_hidden_layers, dropout = self.dropout, return_state = True, return_sequences = True)(encoder_outputs)
                encoder_states = [state]
            if self.cell == "lstm":
                encoder_outputs, state_h, state_c = LSTM(self.num_hidden_layers, dropout = self.dropout, return_state = True, return_sequences = True)(encoder_outputs)
                encoder_states = [state_h,state_c]
            if self.cell == "gru":
                encoder_outputs, state = GRU(self.num_hidden_layers, dropout = self.dropout, return_state = True, return_sequences = True)(encoder_outputs)
                encoder_states = [state]


        decoder_inputs = keras.Input(shape=(None, ))
        decoder_outputs = keras.layers.Embedding(input_dim = self.len_dec_charset + 1, output_dim = self.embedding_size, input_length = self.max_decoder_seq_length)(decoder_inputs)
        decoder_states = list()

        for j in range(self.num_dec_layers):
            if self.cell == "rnn":
                decoder = SimpleRNN(self.num_hidden_layers, dropout = self.dropout, return_sequences = True, return_state = True)
                decoder_outputs, state = decoder(decoder_outputs, initial_state = encoder_states)
                decoder_states += [state]
            if self.cell == "lstm":
                decoder = LSTM(self.num_hidden_layers, dropout = self.dropout, return_sequences = True, return_state = True)
                decoder_outputs, state_h, state_c = decoder(decoder_outputs, initial_state = encoder_states)
                decoder_states += [state_h, state_c]
            if self.cell == "gru":
                decoder = GRU(self.num_hidden_layers, dropout = self.dropout, return_sequences = True, return_state = True)
                decoder_outputs, state = decoder(decoder_outputs, initial_state = encoder_states)

        decoder_dense = Dense(self.len_dec_charset, activation = "softmax")
        decoder_outputs = decoder_dense(decoder_outputs)

        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        self.model = model

    def train(self, encoder_train_input_data, decoder_train_input_data, decoder_train_target_data):

        return self.model.fit(
        x = [encoder_train_input_data, decoder_train_input_data],
        y = decoder_train_target_data,
        batch_size = self.batch_size,
        epochs = self.num_epochs,
        #callbacks = [WandbCallback()]
        )  
        
    def beam_search():
        '''
        Implement Beam Search
        '''
        pass

    def evaluate(val_english, val_indic):
        '''
        Calculates the accuracy of the model on the validation / test set.
        '''
        pass

    def predict(test_english, test_indic):
        '''
        Function to generate predictions of the model.
        '''
        pass