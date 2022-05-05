import tensorflow as tf 
from preprocess import *
from rnn import *
import wandb

def train_wandb(config = None):

    run = wandb.init(config=config, resume=True)

    cfg = wandb.config

    name = f'cell_{cfg.cell}_hidden_{cfg.num_hidden_layers}_encl_{cfg.num_enc_layers}_decl_{cfg.num_dec_layers}_emb_{cfg.embedding_size}_opt_{cfg.optimizer}_drop_{cfg.dropout}'
    

    train_path = "hi.translit.sampled.train.tsv"
    dev_path = "hi.translit.sampled.dev.tsv"
    test_path = "hi.translit.sampled.test.tsv"
    wandb.run.name = name
    wandb.run.save()

  
    cell = cfg.cell
    num_epochs = cfg.num_epochs
    batch_size = cfg.batch_size
    embedding_size = cfg.embedding_size
    num_enc_layers = cfg.num_enc_layers
    num_dec_layers = cfg.num_dec_layers
    num_hidden_layers = cfg.num_hidden_layers
    dropout = cfg.dropout
    optimizer = cfg.optimizer

    #Generate training, validation and test batches (along with paddings, encodings).
    (encoder_train_english, decoder_train_english, decoder_train_indic), (encoder_val_english, decoder_val_english, decoder_val_indic), (val_english, val_indic), (encoder_test_english, decoder_test_english, decoder_test_indic), (english_char_set, indic_char_set, max_seq_len_english_encoder, max_seq_len_indic_decoder), (indic_char_to_idx, indic_idx_to_char), (english_char_to_idx, english_idx_to_char) = preprocess(train_path, dev_path, test_path, batch_size = batch_size)  

    #Create a RNN-model
    rnn_model =  Model(english_char_set, indic_char_set, max_seq_len_english_encoder, max_seq_len_indic_decoder, indic_char_to_idx, indic_idx_to_char, english_char_to_idx, english_idx_to_char, cell = cell, optimizer = optimizer, embedding_size = embedding_size, num_enc_layers = num_enc_layers, num_dec_layers = num_dec_layers, num_hidden_layers = num_hidden_layers, dropout = dropout)
    rnn_model.build_model()

    rnn_model.train(encoder_train_english, decoder_train_english, decoder_train_indic, encoder_val_english, decoder_val_english, decoder_val_indic, num_epochs = num_epochs, batch_size = batch_size)

    word_level_val_acc = rnn_model.evaluate(encoder_val_english, decoder_val_indic)

    wandb.log({'word_level_val_acc': word_level_val_acc})




sweep_config = {
    'method': 'bayes', 
    'metric': {
      'name': 'accuracy',
      'goal': 'maximize'   
              },
    'early_terminate' : {
      'type': 'hyperband',
      'min_iter': 5
    },
    "parameters": {
        "embedding_size": {
            "values": [64,128,256,512]
        },
        "num_enc_layers" :{
            "values" : [1, 3, 5]
        },
        "num_dec_layers": {
            "values": [1, 2, 3]
        },
        "num_hidden_layers": {
            "values": [32, 64, 256, 512]
        },
        "cell": {
            "values": ["rnn", "lstm", "gru"]
        },
        "batch_size": {
            "values": [32, 64, 128]
        },
        "num_epochs": {
            "values": [10, 15]
        },
        "dropout": {
            "values": [0,0.2, 0.3]
        },
        "optimizer": {
            "values": ["adam", "nadam"]
        },
    }
}

project_name = '' #Add project name here
entity = '' #Add username here

wandb.init(project=project_name, entity=entity)

#To add a new agent to an existing sweep, comment next line and directly put sweep_id in wandb.agent
sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)

wandb.agent(sweep_id, project=project_name, function=train_wandb)
