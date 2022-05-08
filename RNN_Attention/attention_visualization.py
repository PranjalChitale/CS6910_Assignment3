import wandb
import numpy as np
import tensorflow as tf
import pandas as pd
from preprocess import *
from rnn_attention import *
import matplotlib.pyplot as plt
from matplotlib import font_manager
from seaborn import heatmap 
import random


font_prop = font_manager.FontProperties(fname='./VesperLibre-Regular.ttf')

project_name = " " #Add project name here
entity = " " #Add username here

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

wandb.init(project = project_name, entity = entity)

train_path = "hi.translit.sampled.train.tsv"
dev_path = "hi.translit.sampled.dev.tsv"
test_path = "hi.translit.sampled.test.tsv"


(encoder_train_english, decoder_train_english, decoder_train_indic), (encoder_val_english, decoder_val_english, decoder_val_indic), (val_english, val_indic), (encoder_test_english, decoder_test_english, decoder_test_indic), (english_char_set, indic_char_set, max_seq_len_english_encoder, max_seq_len_indic_decoder), (indic_char_to_idx, indic_idx_to_char), (english_char_to_idx, english_idx_to_char) = preprocess(train_path, dev_path, test_path, batch_size = 128)  

#Create a RNN-model
rnn_model =  Model(english_char_set, indic_char_set, max_seq_len_english_encoder, max_seq_len_indic_decoder, indic_char_to_idx, indic_idx_to_char, english_char_to_idx, english_idx_to_char, cell = 'gru', optimizer = "adam", embedding_size = 512, num_enc_layers = 1, num_dec_layers = 1, num_hidden_layers = 256, dropout = 0.2)

rnn_model.build_model()

rnn_model.train(encoder_train_english, decoder_train_english, decoder_train_indic, encoder_val_english, decoder_val_english, decoder_val_indic, num_epochs = 10, batch_size = 128)

idx = []

for i in range(0,9):
    n = random.randint(0, len(encoder_train_english)-1)
    idx.append(n)

predictions = []
attentions= []
test_df = load_data(test_path)
test_indic = test_df['indic'].values
rnn_model.inference_setup()

for i in idx:
    input_seq = encoder_test_english[i:i+1]
    output, attn_weights = rnn_model.decode_sequence(input_seq)
    print(test_indic[i].strip(), output.strip())
    
    predictions.append(output)
    attentions.append(attn_weights)

test_df = load_data(test_path)
test_english = test_df['english'].values


fig, ax = plt.subplots(3, 3)
fig.set_size_inches(23, 20)
ax = ax.flatten()

for i in range(0, len(idx)):
    output = predictions[i]
    attention = attentions[i]

    inp_seq = test_english[idx[i]]
    ip_len = len(inp_seq)
    op_len = len(output)

    weights = []
    for j in range(op_len):
        weights.append(attention[j][:ip_len])
    
    ax[i] = heatmap(weights, cbar=True, ax = ax[i], cmap="Blues")
    ax[i].set_xticklabels(inp_seq)
    
    ax[i].set_yticklabels(list(output), fontproperties = font_prop, rotation=0)

plt.savefig('./att_heatmap.jpg')
wandb.log({'Attention_Heatmap' : wandb.Image(fig)})
plt.show()