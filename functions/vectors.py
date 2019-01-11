import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils
from gensim.models import KeyedVectors
from keras.layers import Embedding
from Models.functions.utils import checkFolder
import numpy as np

def vectorFilename(name, embedding_dim, model = 'w2v_model_ug_sg'):
    directory = '/home/rafael/GDrive/Embeddings/'+name+'/'
    checkFolder(directory)
    filename = model+'_'+str(embedding_dim)+'.word2vec'    
    return  directory + filename

def train_vectors(X, name = 'tmp', embedding_dim = 100):

    print('training embeddings...')

    all_x_w2v = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(X)]

    cores = multiprocessing.cpu_count()
    # CBOW sg=0
    # SKIPGRAM sg=1

    model_ug_sg = Word2Vec(sg=1, size=embedding_dim, negative=5, window=5, min_count=0, workers=cores, alpha=0.065, min_alpha=0.065)
    model_ug_sg.build_vocab([x.words for x in all_x_w2v])

    for epoch in range(30):
        model_ug_sg.train(utils.shuffle([x.words for x in all_x_w2v]), total_examples=len(all_x_w2v), epochs=3)
        model_ug_sg.alpha -= 0.002
        model_ug_sg.min_alpha = model_ug_sg.alpha

    vectorName = vectorFilename(name, embedding_dim)
    #model_ug_cbow.save('/content/w2v_model_ug_cbow.word2vec')
    model_ug_sg.save(vectorName)


def create_embeddings(tokenizer, max_num_words, max_seq_length, name='prebuild', embedding_dim=100, filename=None, type=1):

    print('loading embeddings...')
    vectorName = vectorFilename(name, embedding_dim)

    if filename is not None:
        vectorName = filename

    if type == 1:
        model_ug_sg = Word2Vec.load(vectorName)
    else:
        model_ug_sg = KeyedVectors.load_word2vec_format(vectorName, binary=False, unicode_errors="ignore")

    print("Vocab keys", len(model_ug_sg.wv.vocab.keys()))

    embeddings_index = {}
    for w in model_ug_sg.wv.vocab.keys():
        #embeddings_index[w] = np.append(model_ug_cbow.wv[w],model_ug_sg.wv[w])
        embeddings_index[w] = model_ug_sg.wv[w]

    print('Found %s word vectors.' % len(embeddings_index))
    
    num_words = max_num_words
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i >= num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.zeros(embedding_dim)

    print("weights", len(embedding_matrix))

    return Embedding(input_dim=max_num_words, output_dim=embedding_dim,
                     input_length=max_seq_length,
                     weights=[embedding_matrix],
                     trainable=True
                    )
