import numpy as np
import tensorflow as tf
import pandas as pd

def load_data(path):
    '''
    Loads the data from csv file to dataframe.
    '''
    with open(path) as f:
        data = pd.read_csv(f, sep='\t',header=None,names=["indic","english",""],skip_blank_lines=True,index_col=None)
    data = data[data['indic'].notna()]
    data = data[data['english'].notna()]
    data = data[['indic','english']]
    return data

def preprocess(train_path, dev_path, test_path, batch_size):
    '''
    Preprocesses the data to convert it into the form desired for training RNN model.
    '''

    train_df = load_data(train_path)
    val_df = load_data(dev_path)
    test_df = load_data(test_path)

    train_indic = train_df['indic'].values
    train_english = train_df['english'].values
    val_indic = val_df['indic'].values
    val_english = val_df['english'].values
    test_indic = test_df['indic'].values
    test_english = test_df['english'].values


    # "\t" is considered as the "start" character
    # "\n" is considered as the "end" character.

    #We add the above characters to the indic transliterated words.
    train_indic =  "\t" + train_indic + "\n"
    val_indic =  "\t" + val_indic + "\n"
    test_indic =  "\t" + test_indic + "\n"


    #Create character sets for each language
    indic_char_set = set()
    english_char_set = set()

    indic_char_set.add(' ')
    english_char_set.add(' ')
    
    for word_english, word_indic in zip(train_english, train_indic):
        for char in word_english:
            english_char_set.add(char)
        for char in word_indic:
            indic_char_set.add(char)

    english_char_set = sorted(list(english_char_set))
    indic_char_set = sorted(list(indic_char_set))


    #Create empty dicts.
    english_char_to_idx = dict()
    indic_char_to_idx = dict()

    english_idx_to_char = dict()
    indic_idx_to_char = dict()

    #As our character sets don't consider spaces, we assign a special id 0 to space.
    # We will zero-pad the strings to make them of equal length, to support batchwise training.

    english_char_to_idx[" "] = 0
    indic_char_to_idx[" "] = 0

    #Create a mapping of characters to indices    
    for i, char in enumerate(english_char_set):
        english_char_to_idx[char] = i+1

    for i, char in enumerate(indic_char_set):
        indic_char_to_idx[char] = i+1


    #Create a mapping of indices to characters.

    for char, idx in english_char_to_idx.items():
        english_idx_to_char[idx] = char

    for char, idx in indic_char_to_idx.items():
        indic_idx_to_char[idx] = char
    
    #Find the max word length in the indic and english sentences respectively.

    max_seq_len_english_encoder = max([len(word) for word in train_english])
    max_seq_len_indic_decoder = max([len(word) for word in train_indic])

    encoder_train_english = np.zeros((len(train_english), max_seq_len_english_encoder), dtype="float32")
    decoder_train_english = np.zeros((len(train_english), max_seq_len_indic_decoder), dtype="float32")
    decoder_train_indic = np.zeros(
        (len(train_english), max_seq_len_indic_decoder, len(indic_char_set)), dtype="float32"
    )

    encoder_val_english = np.zeros(
        (len(val_english), max_seq_len_english_encoder), dtype="float32"
    )
    decoder_val_english = np.zeros(
        (len(val_english), max_seq_len_indic_decoder), dtype="float32"
    )
    decoder_val_indic = np.zeros(
        (len(val_english), max_seq_len_indic_decoder, len(indic_char_set)), dtype="float32"
    )

    encoder_test_english = np.zeros(
        (len(test_english), max_seq_len_english_encoder), dtype="float32"
    )


    for i, (input_word, target_word) in enumerate(zip(train_english, train_indic)):
        for t, char in enumerate(input_word):
            #Replace character by its index.
            encoder_train_english[i, t] = english_char_to_idx[char]
        #Padding with zeros.
        encoder_train_english[i, t + 1 :] = english_char_to_idx[' ']
        
        for t, char in enumerate(target_word):
            decoder_train_english[i, t] = indic_char_to_idx[char]
            if t > 0:
                # Indic decoder will be ahead by one timestep.
                decoder_train_indic[i, t - 1, indic_char_to_idx[char]-1] = 1.0
        #Padding with zeros.
        decoder_train_english[i, t + 1 :] = indic_char_to_idx[' ']
        decoder_train_indic[i, t :, indic_char_to_idx[' ']-1] = 1.0


    for i, (input_word, target_word) in enumerate(zip(val_english, val_indic)):
        for t, char in enumerate(input_word):
            #Replace character by its index.
            encoder_val_english[i, t] = english_char_to_idx[char]
        encoder_val_english[i, t + 1 :] = english_char_to_idx[' ']
        
        for t, char in enumerate(target_word):
            decoder_val_indic[i, t] = indic_char_to_idx[char]
            if t > 0:
                decoder_val_indic[i, t - 1 :, indic_char_to_idx[char]-1] = 1.0
        decoder_val_indic[i, t + 1 :] =  indic_char_to_idx[' ']
        decoder_val_indic[i, t :, indic_char_to_idx[' ']-1] = 1.0

    for i, input_word in enumerate(test_english):
        for t, char in enumerate(input_word):
            encoder_test_english[i, t] = english_char_to_idx[char]
        encoder_test_english[i, t + 1 :] = english_char_to_idx[' ']


    return (encoder_train_english, decoder_train_english, decoder_train_indic), (encoder_val_english, decoder_val_english, decoder_val_indic), (val_english, val_indic), (encoder_test_english, test_english, test_indic), (english_char_set, indic_char_set, max_seq_len_english_encoder, max_seq_len_indic_decoder), (indic_char_to_idx, indic_idx_to_char)  
    

#Reference : Keras Documentation.
#https://keras.io/examples/nlp/lstm_seq2seq/
#https://stackoverflow.com/questions/54176051/invalidargumenterror-indicesi-0-x-is-not-in-0-x-in-keras