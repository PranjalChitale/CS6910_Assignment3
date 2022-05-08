from tensorflow import keras
from keras.layers import Dense, Input, LSTM, SimpleRNN, GRU, Embedding
from tensorflow.keras.optimizers import Adam,Nadam
import wandb
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, AdditiveAttention
from wandb.keras import WandbCallback
import numpy as np
import pandas as pd


class Model(object):
    def __init__(self, english_char_set, indic_char_set, max_seq_len_english_encoder, max_seq_len_indic_decoder, indic_char_to_idx, indic_idx_to_char, english_char_to_idx, english_idx_to_char, cell ="LSTM", optimizer = "adam", embedding_size = 32, num_enc_layers = 5, num_dec_layers =2, num_hidden_layers = 64, dropout = 0):
        self.len_enc_charset = len(english_char_set)
        self.len_dec_charset = len(indic_char_set)
        self.max_seq_len_english_encoder = max_seq_len_english_encoder
        self.max_seq_len_indic_decoder = max_seq_len_indic_decoder
        self.indic_char_to_idx = indic_char_to_idx
        self.indic_idx_to_char = indic_idx_to_char
        self.english_char_to_idx = english_char_to_idx
        self.english_idx_to_char = english_idx_to_char
        self.cell = cell
        self.embedding_size = embedding_size
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers= num_dec_layers
        self.num_hidden_layers =num_hidden_layers
        self.encoder_model = None
        self.decoder_model = None
        self.model = None
        self.dropout = dropout
        self.num_epochs = None
        self.batch_size = None
        self.optimizer = optimizer
        

    def build_model(self):
        encoder_inputs = Input(shape=(None,), name="encoder_input")
        encoder_outputs = Embedding(self.len_enc_charset, self.embedding_size, name = "encoder_embedding")(encoder_inputs)
        self.enc_layers = []
        self.dec_layers = []
        encoder_states = list()
        for j in range(self.num_enc_layers):
            if self.cell == "rnn":
                encoder = SimpleRNN(self.num_hidden_layers, dropout = self.dropout, return_state = True, return_sequences = True)
                encoder_outputs, state = encoder(encoder_outputs)
                encoder_states.append([state])
                self.enc_layers.append(encoder)
            if self.cell == "lstm":
                encoder = LSTM(self.num_hidden_layers, dropout = self.dropout, return_state = True, return_sequences = True)
                encoder_outputs, state_h, state_c = encoder(encoder_outputs)
                encoder_states.append([state_h,state_c])
                self.enc_layers.append(encoder)
            if self.cell == "gru":
                encoder = GRU(self.num_hidden_layers, dropout = self.dropout, return_state = True, return_sequences = True)
                encoder_outputs, state = encoder(encoder_outputs)
                encoder_states.append([state])
                self.enc_layers.append(encoder)

        self.encoder_model = keras.Model(encoder_inputs,encoder_states)

        decoder_inputs = Input(shape=(self.max_seq_len_indic_decoder, ), name = "decoder_input")
      
        decoder_outputs = Embedding(self.len_dec_charset, self.embedding_size, name = "decoder_embedding")(decoder_inputs)
        decoder_states = list()

        for j in range(self.num_dec_layers):
            if self.cell == "rnn":
                decoder = SimpleRNN(self.num_hidden_layers, dropout = self.dropout, return_sequences = True, return_state = True)
                decoder_outputs, state = decoder(decoder_outputs, initial_state = encoder_states[j])
                decoder_states.append([state])
                self.dec_layers.append(decoder)
            if self.cell == "lstm":
                decoder = LSTM(self.num_hidden_layers, dropout = self.dropout, return_sequences = True, return_state = True)
                decoder_outputs, state_h, state_c = decoder(decoder_outputs, initial_state = encoder_states[j])
                decoder_states.append([state_h, state_c])
                self.dec_layers.append(decoder)
            if self.cell == "gru":
                decoder = GRU(self.num_hidden_layers, dropout = self.dropout, return_sequences = True, return_state = True)
                decoder_outputs, state = decoder(decoder_outputs, initial_state = encoder_states[j])
                decoder_states.append([state])
                self.dec_layers.append(decoder)

        decoder_attn = AdditiveAttention(name="attention_layer")
        decoder_concat = Concatenate(name="concatenate_layer")
        cont_vec, attn_wts = decoder_attn([decoder_outputs, encoder_outputs],return_attention_scores=True)
        decoder_outputs = decoder_concat([decoder_outputs,cont_vec])
        
        dec_dense =Dense(self.len_dec_charset, activation="softmax", name="dense_layer")
        dec_pred = dec_dense(decoder_outputs)
            
        
        model = keras.Model([encoder_inputs, decoder_inputs], dec_pred)

        model.compile(
            optimizer=self.optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )

        self.model = model


    def train(self, encoder_train_english, decoder_train_english, decoder_train_indic, encoder_val_english, decoder_val_english, decoder_val_indic, num_epochs =10, batch_size = 64):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model.fit(
        x = [encoder_train_english, decoder_train_english],
        y = decoder_train_indic,
        validation_data = ([encoder_val_english, decoder_val_english], decoder_val_indic),
        batch_size = self.batch_size,
        epochs = self.num_epochs,
        callbacks = [WandbCallback()]
        )  

    def inference_setup(self):

        #First input layer will be for encoder
    
        encoder_inputs = self.model.input[0]

        #First embedding layer will be of encoder.

        enc_embed_layer = self.model.get_layer('encoder_embedding')

        encoder_outputs = enc_embed_layer(encoder_inputs)

        encoder_states = []

        if self.cell == 'rnn':
            for i in range(self.num_enc_layers):
                encoder_outputs, state_h = self.enc_layers[i](encoder_outputs)
                encoder_states += [state_h] 
        elif self.cell == 'lstm':
            for i in range(self.num_enc_layers):
                encoder_outputs, state_h, state_c = self.enc_layers[i](encoder_outputs)
                encoder_states += [state_h, state_c]   
        elif self.cell == 'gru':
            for i in range(self.num_enc_layers):
                encoder_outputs, state_h = self.enc_layers[i](encoder_outputs)
                encoder_states += [state_h] 

        self.encoder_model = keras.Model(encoder_inputs, encoder_states + [encoder_outputs])


        decoder_inputs = self.model.input[1]    
        dec_embed_layer = self.model.get_layer('decoder_embedding')
        decoder_outputs = dec_embed_layer(decoder_inputs)

        dec_states = []
        dec_input_states = []
        
        if self.cell == 'lstm' :
            j=0
            for i in range(self.num_dec_layers):
                dec_input_states += [Input(shape=(self.num_hidden_layers, )) , Input(shape=(self.num_hidden_layers, ))]
                decoder_outputs, state_h, state_c = self.dec_layers[i](decoder_outputs, initial_state = dec_input_states[i+j:i+j+2])
                dec_states += [state_h , state_c]
                j += 1

        else:
            for i in range(self.num_dec_layers):
                dec_input_states += [Input(shape=(self.num_hidden_layers,))]
                decoder_outputs, state_h = self.dec_layers[i](decoder_outputs, initial_state = dec_input_states[i])
                dec_states += [state_h]

        attention_layer = self.model.get_layer('attention_layer')

        attention_input = Input(shape=(self.max_seq_len_english_encoder,self.num_hidden_layers))   

        context_vector, alphas = attention_layer([decoder_outputs, attention_input], return_attention_scores=True)
    
        concat_layer = self.model.get_layer('concatenate_layer')

        decoder_outputs = concat_layer([decoder_outputs, context_vector])


        # Dense layer
        decoder_dense = self.model.get_layer('dense_layer')

        decoder_outputs = decoder_dense(decoder_outputs)

        # Decoder model
        self.decoder_model = keras.Model(
            [decoder_inputs] + dec_input_states + [attention_input], [decoder_outputs] + dec_states + [alphas])

    def decode_sequence(self, input_seq):
        # Setup the encoder-decoder models.
        # Due to some issues with shapes and broadcasting, batch decoding code from vanilla RNN wasn't working here.
        # Therefore, sentence by sentence decoding.

        self.inference_setup()

        #Get the encoder output, attention and states
        enc_states = self.encoder_model.predict(input_seq)
        attention_input = enc_states[-1]

        enc_states = enc_states[:-1]
        
        prev_seq = np.zeros((1, 1)) 
        prev_seq[0, 0] = self.indic_char_to_idx["\t"]
        
        attention_weights = []
        stop_condition = False
        decoded_sentence = ""

        while not stop_condition:
            output_tokens = self.decoder_model.predict([prev_seq] + enc_states + [attention_input])
            sampled_token_index = np.argmax(output_tokens[0][0, -1, :])
            sampled_char = self.indic_idx_to_char[sampled_token_index]
            decoded_sentence += sampled_char

            if sampled_char == "\n" or len(decoded_sentence) > self.max_seq_len_indic_decoder:
                stop_condition = True

            prev_seq = np.zeros((1, 1))
            prev_seq[0, 0] = sampled_token_index

            enc_states = output_tokens[1:-1]
            attention_weights.append(output_tokens[-1][0][0])
            
        return decoded_sentence, attention_weights



