import argparse
import tensorflow as tf 
from preprocess import *
from rnn import *

#Define the Command Line Arguments
parser = argparse.ArgumentParser(description='Set the directory paths, hyperparameters of the model.')
parser.add_argument('--train_path', type=str, default='dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv', help='Path of the train data directory.')
parser.add_argument('--dev_path', type=str, default='dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv', help='Path of the Validation data directory.')
parser.add_argument('--test_path', type=str, default='dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv', help='Path of the test data directory')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of Epochs')
parser.add_argument('--embedding_size', type=int, help='Size of the embedding layer to be used', default=128)
parser.add_argument('--dropout', type=float, default=0, help='Dropout Rate')


#Parse the arguments
args = parser.parse_args()
train_path = args.train_path
dev_path = args.dev_path
test_path = args.test_path
batch_size = args.batch_size
learning_rate = args.learning_rate
dense_neurons = args.embedding_size
dropout = args.dropout
num_epochs = args.num_epochs


#Generate training, validation and test batches (along with paddings, encodings).
(encoder_train_english, decoder_train_english, decoder_train_indic), (encoder_val_english, decoder_val_english, decoder_val_indic), (val_english, val_indic), (encoder_test_english, test_english, test_indic), (english_char_set, indic_char_set, max_seq_len_english_encoder, max_seq_len_indic_decoder), (indic_char_to_idx, indic_idx_to_char) = preprocess(train_path, dev_path, test_path, batch_size = 64) 

rnn_model =  Model(english_char_set, indic_char_set, max_seq_len_english_encoder, max_seq_len_indic_decoder, indic_char_to_idx, indic_idx_to_char, cell ="LSTM", num_epochs =10, batch_size = 64, embedding_size = 32, num_enc_layers = 5, num_dec_layers =2, num_hidden_layers = 3, dropout = 0)

rnn_model.build()

rnn_model.train(encoder_train_english, decoder_train_english, decoder_train_indic)