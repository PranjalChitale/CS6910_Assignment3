from tensorflow import keras
from keras.layers import SimpleRNN,LSTM,GRU,Embedding,Dense,Dropout,Input
from tensorflow.keras.optimizers import Adam,Nadam
import wandb
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
        encoder_inputs = Input(shape=(None,))
        encoder_context = Embedding(input_dim = self.len_enc_charset + 1, output_dim = self.embedding_size, input_length = self.max_seq_len_indic_decoder )(encoder_inputs)
        encoder_outputs = encoder_context
        encoder_states = list()
        for j in range(self.num_enc_layers):
            if self.cell == "rnn":
                encoder_outputs, state = SimpleRNN(self.num_hidden_layers, dropout = self.dropout, return_state = True, return_sequences = True)(encoder_outputs)
                encoder_states += [state]
            if self.cell == "lstm":
                encoder_outputs, state_h, state_c = LSTM(self.num_hidden_layers, dropout = self.dropout, return_state = True, return_sequences = True)(encoder_outputs)
                encoder_states += [state_h,state_c]
            if self.cell == "gru":
                encoder_outputs, state = GRU(self.num_hidden_layers, dropout = self.dropout, return_state = True, return_sequences = True)(encoder_outputs)
                encoder_states += [state]

        self.encoder_model = keras.Model(encoder_inputs,encoder_states)

        decoder_inputs = Input(shape=(None, ))
      
        decoder_outputs = Embedding(input_dim = self.len_dec_charset + 1, output_dim = self.embedding_size, input_length = self.max_seq_len_indic_decoder)(decoder_inputs)
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


        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

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

    def get_layer_index(self):

        enc_embed_index, dec_embed_index = -1, -1
        enc_layers_index = [] 
        dec_layers_index = []
        dense_layer_index = -1
        
        for i, layer in enumerate(self.model.layers):
            # For the final Dense Layer
            if "dense" in layer.name : 
                dense_layer_index = i

            # Embedding layer
            if "embedding" in layer.name:
                if enc_embed_index == -1 : 
                    #If this is first embedding layer, it will be encoder.
                    enc_embed_index = i
                else: 
                    #Otherwise decoder
                    dec_emb_index = i

            # RNN 
            if self.cell in layer.name:
                if len(enc_layers_index) < self.num_enc_layers:
                    enc_layers_index.append(i)
                else:
                    dec_layers_index.append(i)

        return enc_embed_index, dec_emb_index, enc_layers_index, dec_layers_index, dense_layer_index

        
    def inference_setup(self):
    
        enc_embed_index, dec_emb_index, enc_layers_index, dec_layers_index, dense_layer_index  = self.get_layer_index()

        decoder_inputs = self.model.input[1]  # input 0 is encoder, input[1] will correspond to decoder.

        #Get the input to the embedding layer and generate the ourput.
        decoder_outputs =  self.model.layers[dec_emb_index](decoder_inputs)

  
        decoder_states = []
        dec_input_states = []


        for dec in range(len(dec_layers_index)):
            #Iterate through the decoder layers.
        
            if self.cell == "rnn" or self.cell == "gru":
                #Create an input for states.
                state = keras.Input(shape = (self.num_hidden_layers, ))
                current_states_inputs = [state]
                decoder_outputs, state = self.model.layers[dec_layers_index[dec]](decoder_outputs, initial_state = current_states_inputs)
                decoder_states += [state]

            elif self.cell == "lstm":
                state_h_dec, state_c_dec = keras.Input(shape = (self.num_hidden_layers,)),  keras.Input(shape = (self.num_hidden_layers,))
                current_states_inputs = [state_h_dec, state_c_dec]
                decoder_outputs, state_h_dec,state_c_dec = self.model.layers[dec_layers_index[dec]](decoder_outputs, initial_state = current_states_inputs)
                decoder_states += [state_h_dec, state_c_dec]
            
            dec_input_states += current_states_inputs

        # Dense layer
        decoder_dense = self.model.layers[dense_layer_index]
        decoder_outputs = decoder_dense(decoder_outputs)

        # Decoder model
        self.decoder_model = keras.Model(
            [decoder_inputs] + dec_input_states, [decoder_outputs] + decoder_states
        )

    def inference(self, inp_seq):
        enc_states = self.encoder_model.predict(inp_seq)
        prev_seq = np.zeros((inp_seq.shape[0],1))
        prev_seq[:,0] = self.indic_char_to_idx["\t"]
        
        states = []
        
        if self.cell == "LSTM":
            for c in range(self.num_dec_layers):
                states += [enc_states[0],enc_states[1]]
                
        else:
            for c in range(self.num_dec_layers):
                states += [enc_states]
                
        pred = np.zeros((inp_seq.shape[0],self.max_seq_len_indic_decoder))
        
        for i in range(self.max_seq_len_indic_decoder):
            output = self.decoder_model.predict(tuple([prev_seq]+states))
            pred[:,i] = np.argmax(output[0][:,-1,:],axis=1)
            prev_seq[:,0] = pred[:,i]
            states = output[1:]
            
        return pred

    def evaluate(self, english, indic):
        '''
        Calculates the accuracy of the model on the validation / test set.
        '''
        
        correct = 0

        self.inference_setup()

        pred  = self.inference(english)

        for i,pr in enumerate(pred):
            flag = 1
            for j,ch in enumerate(pr):
                if ch != np.argmax(indic[i,j,:]):
                    flag = 0
                    break
                if ch == self.indic_char_to_idx["\n"]:
                    break
                    
            if flag==1:
                correct+=1
                
                
        return (correct/len(pred))  

    
    def pred_2_csv(self, english, true, indic):
        '''
        Generates a CSV file containing the english input, 
        true indic transliteration and prediction of the model.
        '''
        
        true_english = []
        true_indic = []
        pred_indic = []
        self.inference_setup()

        pred  = self.inference(english)

        for i,pr in enumerate(pred):
            ip_eng = ""
            for ch in english[i]:
                if self.english_idx_to_char[ch] == " ":
                    break
                else:
                    ip_eng += self.english_idx_to_char[ch]
            true_english.append(ip_eng)


            ip_indic = ""  
            
            for ch in true[i,1:]:
                if self.indic_idx_to_char[ch] == "\n":
                    break
                else:
                    ip_indic += self.indic_idx_to_char[ch]
            true_indic.append(ip_indic)

            op_indic = ""
            for j,ch in enumerate(pr):
                
                if self.indic_idx_to_char[ch] == "\n":
                    break
                else:
                    op_indic += self.indic_idx_to_char[ch]
                
            pred_indic.append(op_indic)
            
        df = pd.DataFrame(list(zip(true_english, pred_indic, true_indic)), columns =['English', 'Prediction', 'True'])

        df.to_csv('predictions_vanilla.csv')
        
        return df
