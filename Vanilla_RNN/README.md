# CS6910_Assignment3
Use recurrent neural networks to build a transliteration system.
---


### How to pass the hyperparameters configuration as command-line arguments to main.py  ?
```
--train_path # Path of the train data directory.
--dev_path # Path of the Validation data directory..
--test_path' #Path of the test data directory
--batch_size #Batch size
--optimizer #Optimizer
--num_epochs #Number of Epochs to train for.
--num_enc_layers #Number of Encoder Layers
--num_dec_layers #Number of Decoder Layers
--num_hidden_layers #Number of Hidden Layers
--embedding_size #Size of the embedding layer to be used
--dropout #Dropout Rate, 0 indicates no dropout.
--cell #Cell type
```


Example :-
python main.py --num_enc_layers 5  --num_dec_layers 5  --num_hidden_layers 128 \
--embedding_size 512 --cell 'lstm' --batch_size 64 --num_epochs 20 \
--optimizer 'adam' --dropout 0.2 \

### To generate a CSV containing predictions of the model :-
1. Run preprocessing.
2. Run :-
- Build the model
- Train the model
- And then use
- rnn_model.pred_2_csv(encoder_test_english, decoder_test_english, decoder_test_indic)

### Use WANDB sweep :-
In wandb_sweep_vanilla.py

- First Set your project name and username
wandb.init(project=project_name, entity=entity)

- To add a new agent to an existing sweep, comment next line and directly put sweep_id in wandb.agent
sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)
wandb.agent(sweep_id, project=project_name, function=train_wandb)

- Change the config_sweep as per the need, to sweep using different strategy.
