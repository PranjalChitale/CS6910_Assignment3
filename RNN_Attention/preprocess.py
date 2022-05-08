import numpy as np
import tensorflow as tf
import pandas as pd

def load_data(path):
    with open(path) as f:
        data = pd.read_csv(f, sep='\t',header=None,names=["indic","english",""],skip_blank_lines=True,index_col=None)
    data = data[data['indic'].notna()]
    data = data[data['english'].notna()]
    data = data[['indic','english']]
    return data

def preprocess(train_path, dev_path, test_path, batch_size):

    train_df = load_data(train_path)
    val_df = load_data(dev_path)
    test_df = load_data(test_path)

    train_df['indic'] = train_df['indic'].str.replace(" ","")
    train_df['english'] = train_df['english'].str.replace(" ","")
    val_df['indic'] = val_df['indic'].str.replace(" ","")
    val_df['english'] = val_df['english'].str.replace(" ","")
    test_df['indic'] = test_df['indic'].str.replace(" ","")
    test_df['english'] = test_df['english'].str.replace(" ","")


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
    # We will pad the strings with spaces to make them of equal length, to support batchwise training.

    english_char_to_idx[" "] = 0
    indic_char_to_idx[" "] = 0

    #Create a mapping of characters to indices    
    for i, char in enumerate(english_char_set):
        english_char_to_idx[char] = i

    for i, char in enumerate(indic_char_set):
        indic_char_to_idx[char] = i


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
    decoder_test_english = np.zeros(
        (len(test_english), max_seq_len_indic_decoder), dtype="float32"
    )
    decoder_test_indic = np.zeros(
        (len(test_english), max_seq_len_indic_decoder, len(indic_char_set)), dtype="float32"
    )

    print(encoder_train_english.shape, "ENC Train Eng")
    print(decoder_train_english.shape, "DEC Train Eng")
    print(decoder_train_indic.shape, "DEC Train Indic")
    print(encoder_val_english.shape, "ENC Val Eng")
    print(decoder_val_english.shape, "DEC Val Eng")
    print(decoder_val_indic.shape, "DEC Val Eng")
    print(encoder_test_english.shape, "ENC Test Eng")
    print(decoder_test_english.shape, "DEC Test Eng")
    print(decoder_test_indic.shape, "DEC Test Eng")
  

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
                decoder_train_indic[i, t - 1, indic_char_to_idx[char]] = 1.0
        #Padding with spaces.
        decoder_train_english[i, t + 1 :] = indic_char_to_idx[' ']
        decoder_train_indic[i, t :, indic_char_to_idx[' ']] = 1.0


    for i, (input_word, target_word) in enumerate(zip(val_english, val_indic)):
        for t, char in enumerate(input_word):
            #Replace character by its index.
            encoder_val_english[i, t] = english_char_to_idx[char]
        #Padding with zeros.
        encoder_val_english[i, t + 1 :] = english_char_to_idx[' ']
        
        for t, char in enumerate(target_word):
            decoder_val_english[i, t] = indic_char_to_idx[char]
            if t > 0:
                # Indic decoder will be ahead by one timestep.
                decoder_val_indic[i, t - 1, indic_char_to_idx[char]] = 1.0
        #Padding with spaces.
        decoder_val_english[i, t + 1 :] = indic_char_to_idx[' ']
        decoder_val_indic[i, t :, indic_char_to_idx[' ']] = 1.0

    for i, (input_word, target_word) in enumerate(zip(test_english, test_indic)):
        for t, char in enumerate(input_word):
            #Replace character by its index.
            encoder_test_english[i, t] = english_char_to_idx[char]
        #Padding with spaces.
        encoder_test_english[i, t + 1 :] = english_char_to_idx[' ']
        
        for t, char in enumerate(target_word):
            decoder_test_english[i, t] = indic_char_to_idx[char]
            if t > 0:
                # Indic decoder will be ahead by one timestep.
                decoder_test_indic[i, t - 1, indic_char_to_idx[char]] = 1.0
        #Padding with spaces.
        decoder_test_english[i, t + 1 :] = indic_char_to_idx[' ']
        decoder_test_indic[i, t :, indic_char_to_idx[' ']] = 1.0


    return (encoder_train_english, decoder_train_english, decoder_train_indic), (encoder_val_english, decoder_val_english, decoder_val_indic), (val_english, val_indic), (encoder_test_english, decoder_test_english, decoder_test_indic), (english_char_set, indic_char_set, max_seq_len_english_encoder, max_seq_len_indic_decoder), (indic_char_to_idx, indic_idx_to_char), (english_char_to_idx, english_idx_to_char)
    

#Reference : Keras Documentation.
#https://keras.io/examples/nlp/lstm_seq2seq/
#https://stackoverflow.com/questions/54176051/invalidargumenterror-indicesi-0-x-is-not-in-0-x-in-keras