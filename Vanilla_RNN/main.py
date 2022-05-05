import argparse
import tensorflow as tf 
from preprocess import *
from rnn import *

#Define the Command Line Arguments
parser = argparse.ArgumentParser(description='Set the directory paths, hyperparameters of the model.')
parser.add_argument('--train_path', type=str, default='dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv', help='Path of the train data directory.')
parser.add_argument('--dev_path', type=str, default='dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv', help='Path of the Validation data directory.')
parser.add_argument('--test_path', type=str, default='dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv', help='Path of the test data directory')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of Epochs')
parser.add_argument('--num_enc_layers', type=int, default=5, help='Number of Encoder Layers')
parser.add_argument('--num_dec_layers', type=int, default=5, help='Number of Decoder Layers')
parser.add_argument('--num_hidden_layers', type=int, default=5, help='Number of Hidden Layers')
parser.add_argument('--embedding_size', type=int, help='Size of the embedding layer to be used', default=128)
parser.add_argument('--dropout', type=float, default=0, help='Dropout Rate')
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

word_level_val_acc = rnn_model.evaluate(encoder_val_english, decoder_val_indic)

print(word_level_val_acc)

df = rnn_model.pred_2_csv(encoder_test_english, decoder_test_english, decoder_test_indic)
df.head()