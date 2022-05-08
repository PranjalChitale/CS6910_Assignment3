import argparse
import tensorflow as tf 
from preprocess import *
from rnn_attention import *

#Define the Command Line Arguments
parser = argparse.ArgumentParser(description='Set the directory paths, hyperparameters of the model.')
parser.add_argument('--train_path', type=str, default='dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv', help='Path of the train data directory.')
parser.add_argument('--dev_path', type=str, default='dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv', help='Path of the Validation data directory.')
parser.add_argument('--test_path', type=str, default='dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv', help='Path of the test data directory')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer')
parser.add_argument('--num_epochs', type=int, default=15, help='Number of Epochs')
parser.add_argument('--num_enc_layers', type=int, default=1, help='Number of Encoder Layers')
parser.add_argument('--num_dec_layers', type=int, default=1, help='Number of Decoder Layers')
parser.add_argument('--num_hidden_layers', type=int, default=256, help='Number of Hidden Layers')
parser.add_argument('--embedding_size', type=int, help='Size of the embedding layer to be used', default=512)
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout Rate')
parser.add_argument('--cell', type=str, default="lstm", help='Cell type')

#Parse the arguments
args = parser.parse_args()
train_path = args.train_path
dev_path = args.dev_path
test_path = args.test_path
batch_size = args.batch_size
optimizer = args.optimizer
num_epochs = args.num_epochs
embedding_size = args.embedding_size
num_dec_layers = args.num_dec_layers
num_enc_layers = args.num_enc_layers
num_hidden_layers = args.num_hidden_layers
dropout = args.dropout
cell = args.cell

#Generate training, validation and test batches (along with paddings, encodings).
(encoder_train_english, decoder_train_english, decoder_train_indic), (encoder_val_english, decoder_val_english, decoder_val_indic), (val_english, val_indic), (encoder_test_english, decoder_test_english, decoder_test_indic), (english_char_set, indic_char_set, max_seq_len_english_encoder, max_seq_len_indic_decoder), (indic_char_to_idx, indic_idx_to_char), (english_char_to_idx, english_idx_to_char) = preprocess(train_path, dev_path, test_path, batch_size = batch_size)  

#Create a RNN-model
rnn_model =  Model(english_char_set, indic_char_set, max_seq_len_english_encoder, max_seq_len_indic_decoder, indic_char_to_idx, indic_idx_to_char, english_char_to_idx, english_idx_to_char, cell = cell, optimizer = optimizer, embedding_size = embedding_size, num_enc_layers = num_enc_layers, num_dec_layers = num_dec_layers, num_hidden_layers = num_hidden_layers, dropout = dropout)
rnn_model.build_model()

rnn_model.train(encoder_train_english, decoder_train_english, decoder_train_indic, encoder_val_english, decoder_val_english, decoder_val_indic, num_epochs = num_epochs, batch_size = batch_size)

test_df = load_data(test_path)
test_indic = test_df['indic'].values
ctr = 0

test_df = load_data(test_path)
test_indic = test_df['indic'].values
test_eng = test_df['english'].values

ctr = 0

true = []
pred = []
eng = []

for i in range(len(test_indic)):
    input_seq = encoder_test_english[i:i+1]
    output, attn_weights = rnn_model.decode_sequence(input_seq)
    print(i, ctr)
    if test_indic[i].strip() == output.strip():
        ctr+=1
    eng.append(test_eng.strip())
    true.append(test_indic[i].strip())
    pred.append(output.strip())
    


print(ctr/len(test_indic))

dic_prd = {'English':eng, 'true_indic':true, 'pred_indic':pred}

df = pd.DataFrame(dic_prd) 
    
# saving the dataframe 
df.to_csv('Predictions_Attention.csv') 

