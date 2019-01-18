# -*- coding: utf-8 -*-
"""
CNN model for text classification
This implementation is based on the original paper of Yoon Kim [1].

# References
- [1] [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

"""
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

from keras.layers import Activation, Input, Dense, Flatten, Dropout, Embedding, RepeatVector
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.merge import concatenate, add
from keras.layers.recurrent import GRU
from keras import regularizers
from keras.models import Model, Sequential



def build_cnn1(embedding_layer=None, num_words=None,
              embedding_dim=None, filter_sizes=[3,4,5],
              feature_maps=[100,100,100], max_seq_length=100, dropout_rate = None, 
              n_classes = 2, dense_units = None, pool_size = [1,1,1], strides = [1,1,1], layers = False):

    __version__ = 'b3/0.0.2'

    """
    Building a CNN for text classification
    
    Arguments:
        embedding_layer : If not defined with pre-trained embeddings it will be created from scratch
        num_words       : Maximal amount of words in the vocabulary
        embedding_dim   : Dimension of word representation
        filter_sizes    : An array of filter sizes per channel
        feature_maps    : Defines the feature maps per channel
        max_seq_length  : Max length of sequence
        dropout_rate    : If defined, dropout will be added after embedding layer & concatenation
        
    Returns:
        Model           : Keras model instance
    """
    
    # Checks
    if len(filter_sizes)!=len(feature_maps):
        raise Exception('Please define `filter_sizes` and `feature_maps` with the same length.')
    if not embedding_layer and (not num_words or not embedding_dim):
        raise Exception('Please define `num_words` and `embedding_dim` if you not use a pre-trained embedding')
    
    print('Creating CNN %s' % __version__)
    print('#############################################')
    print('Embedding:    %s pre-trained embedding' % ('using' if embedding_layer else 'no'))
    print('Vocabulary size: %s' % num_words)
    print('Embedding dim: %s' % embedding_dim)
    print('Filter sizes: %s' % filter_sizes)
    print('Feature maps: %s' % feature_maps)
    print('Max sequence: %i' % max_seq_length)
    #print('Dropout rate: %f' % dropout_rate)
    print('#############################################')  
    
    if embedding_layer is None:
        embedding_layer = Embedding(input_dim=num_words, output_dim=embedding_dim,
                                    input_length=max_seq_length,
                                    weights=None,
                                    trainable=True
                                   )
    
    channels = []
    x_in = Input(shape=(max_seq_length,), dtype='int32')    
    
    emb_layer = embedding_layer(x_in)

    #if dropout_rate:
        #emb_layer  = Dropout(dropout_rate)(emb_layer)
    
    for ix in range(len(filter_sizes)):        
        x = create_channel(emb_layer, filter_sizes[ix], feature_maps[ix], pool_size[ix], strides[ix])
        channels.append(x)        
    
    # Concatenate all channels
    if len(filter_sizes) > 1:
        x = concatenate(channels)
    else:
        x = x

    #x = MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
    x = GlobalMaxPooling1D()(x)
    #x = Flatten()(x)
    
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)

    if dense_units is not None:        
        for d_units in dense_units:            
            x = Dense(units = d_units, activation = 'relu')(x)

    x = Dense(n_classes, activation='softmax')(x)
    
    return Model(inputs=x_in, outputs=x)
    
def create_channel(x, filter_size, feature_map, pool_size, strides):
    """
    Creates a layer working channel wise
    """
    x = Conv1D(feature_map, kernel_size=filter_size, activation='relu', strides=strides,
               padding='same', kernel_regularizer=regularizers.l2(0.03))(x)
    x = MaxPooling1D(pool_size=pool_size, strides=strides, padding='valid')(x)
    # x = GlobalMaxPooling1D()(x)
    """
    x = Conv1D(feature_map, kernel_size=filter_size, activation='relu', strides=strides,
               padding='same', kernel_regularizer=regularizers.l2(0.03))(x)
    x = MaxPooling1D(pool_size=pool_size, strides=strides, padding='valid')(x)    
    """
    # x = Flatten()(x)
    return x



# SIMPLE CNN


# DNN MODEL


def build_dnn1(num_words):

    model = Sequential()

    model.add(Dense(units=int(num_words / 2), activation='relu', input_dim=num_words))

    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))

    model.add(Dense(units=1, activation='sigmoid'))

    return model
