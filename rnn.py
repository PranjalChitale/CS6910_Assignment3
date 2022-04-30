from tensorflow import keras
from keras.layers import SimpleRNN,LSTM,GRU,Embedding,Dense,Dropout,Input
from tensorflow.keras.optimizers import Adam,Nadam
import wandb
from wandb.keras import WandbCallback
import numpy as np


class Model(object):
    def __init__(self, english_char_set, indic_char_set, max_seq_len_english_encoder, max_seq_len_indic_decoder, indic_char_to_idx, indic_idx_to_char, cell ="LSTM", num_epochs =10, optimizer = "adam", batch_size = 64, embedding_size = 32, num_enc_layers = 5, num_dec_layers =2, num_hidden_layers = 3, dropout = 0):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.len_enc_charset = len(english_char_set)
        self.len_dec_charset = len(indic_char_set)
        self.max_seq_len_english_encoder = max_seq_len_english_encoder
        self.max_seq_len_indic_decoder = max_seq_len_indic_decoder
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
        self.optimizer = optimizer
        self.dropout = dropout
        

    def build_model(self):
        encoder_inputs = Input(shape=(None,))
        encoder_context = Embedding(input_dim = self.len_enc_charset + 1, output_dim = self.embedding_size, input_length = self.max_seq_len_indic_decoder )(encoder_inputs)
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

        self.encoder_model = keras.Model(encoder_inputs,encoder_states)

        decoder_inputs = keras.Input(shape=(None, ))
        #state_inputs = []
        decoder_outputs = keras.layers.Embedding(input_dim = self.len_dec_charset + 1, output_dim = self.embedding_size, input_length = self.max_seq_len_indic_decoder)(decoder_inputs)
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
                decoder_states += [state]

        decoder_dense = Dense(self.len_dec_charset, activation = "softmax")
        decoder_outputs = decoder_dense(decoder_outputs)

        self.decoder_model = keras.Model(
        [decoder_inputs] + encoder_states, [decoder_outputs] + decoder_states
        )


        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        model.compile(
            optimizer=self.optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )

        self.model = model

    def train(self, encoder_train_input_data, decoder_train_input_data, decoder_train_target_data, encoder_val_english, decoder_val_english, decoder_val_indic):

        return self.model.fit(
        x = [encoder_train_input_data, decoder_train_input_data],
        y = decoder_train_target_data,
        validation_data = ([encoder_val_english, decoder_val_english], decoder_val_indic),
        batch_size = self.batch_size,
        epochs = self.num_epochs,
        callbacks = [WandbCallback()]
        )  
        
    def beam_search(self, inp_seq, beam_size):
        '''
        Implement Beam Search
        Beam size = 1 : Greedy Search.
        '''
        if beam_size == 1:
            #TODO : Greedy Search
            pass
        else:
            #TODO : Implement beam search
            pass

    def evaluate(self, val_english, val_indic):
        '''
        Calculates the accuracy of the model on the validation / test set.
        '''
        pass

    def predict(self, test_english, test_indic):
        '''
        Function to generate predictions of the model.
        '''

        pass