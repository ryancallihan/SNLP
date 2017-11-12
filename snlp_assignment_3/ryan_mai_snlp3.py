"""
SNLP Assignment 3
Submission from:
Ryan Callihan | 4076629
Mai Mayeg | 4053079
"""
import gensim
import numpy as np
import pandas as pd
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, SimpleRNN, LSTM, Bidirectional, Activation
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale
from tqdm import tqdm
LabeledSentence = gensim.models.doc2vec.LabeledSentence


def index_text(texts, tokenizer):
    print("INDEXING>>>>")
    return tokenizer.texts_to_sequences(texts)


def vectorize_text(texts, tokenizer):
    """
    Takes a list of texts and returns a matrix of "bag of word" vectors.
    Each row represents a text with a vector of length: number of types.
    includes a 1 for each type in the text.
    :param texts:
    :param tokenizer: # Keras tokenizer object
    :return:
    """
    print("VECTORIZING>>>>")

    indexed = tokenizer.texts_to_sequences(texts)
    num_items = len(tokenizer.word_index)
    vectors = []
    for index in indexed:
        index = index
        vec = np.zeros(num_items, dtype=np.int).tolist()
        for idx in index:
            vec[idx - 1] = 1
        vectors.append(vec)
    return vectors


def preprocessing(text, cachedStopWords):
    """
    remove punctuations and stop words
    :param text: pandas data frame
    :param cachedStopWords:
    :return: return same pandas data frame but clean
    """
    for sents in text:
        cleantext = sents.translate(str.maketrans('', '', string.punctuation))
        removedstop = ' '.join([word for word in cleantext.lower().split() if word not in cachedStopWords])
        text = text.replace(sents, removedstop)
    return text


def labelize(text, label_type):
    """
    labelize the text for word2vec
    :param text: a numpy array
    :param label_type: a string
    :return:
    """
    labelized = []
    for i, v in tqdm(enumerate(text)):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


def train_word2vec(x_train, x_test):
    """
    generates a vector ready to train word2vec model
    :param x_train:
    :param x_test:
    :return:
    """
    word2vec_list = []
    for sent in x_train:
        word2vec_list.append(sent)
    for sent in x_test:
        word2vec_list.append(sent)
    word2vec_array = np.array(word2vec_list)
    return word2vec_array


def sentence_vectorizer(tokens, w2v, tfidf, size):
    """
    takes a tokens and creates a vector based on word2vec model
    :param tokens:
    :param w2v:
    :param tfidf:
    :param size:
    :return:
    """
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def model(input_dim, output):
    """
    Feed forward neural net classifier
    :param input_dim:
    :param output:
    :return:
    """
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=input_dim))
    model.add(Dense(output, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # model = Sequential()
    # model.add(Dense(512, input_dim=input_dim))
    # model.add(Dropout(0.2))
    # for i in range(2):
    #     model.add(Dense(256))
    #     model.add(Activation("tanh"))
    #     model.add(Dropout(0.1))
    # model.add(Dense(output, activation='softmax'))
    # opt = SGD(lr=0.01)
    # model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    # model.compile(loss='sparse_categorical_crossentropy',
    #               optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == "__main__":

    print("READING DATA>>>>")

    cachedStopWords = stopwords.words("german")
    cachedStopWords.append('dbbahn')
    cachedStopWords.append('ja')

    train_data = pd.read_csv('data/train_v1.4.tsv', sep='\t', header=None)
    test_data = pd.read_csv('data/test_TIMESTAMP1.tsv', sep='\t', header=None)

    x_train = preprocessing(train_data[1], cachedStopWords)
    x_test = preprocessing(test_data[1], cachedStopWords)

    print("TRAINING W2V>>>>")
    word2vec_train = train_word2vec(x_train, x_test)
    word2vec_train = labelize(word2vec_train, "TRAIN")

    print("BUILDING FROM W2V>>>>")
    w2v = gensim.models.Word2Vec(size=200, min_count=5)
    w2v.build_vocab([x for x in x_train])
    w2v.train([x for x in x_train], total_examples=w2v.corpus_count, epochs=w2v.iter)

    print('BUILDING TFIDF >>>>')
    tfidf_vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
    tfidf_vectorizer.fit_transform([x for x in x_train])
    tfidf = dict(zip(tfidf_vectorizer.get_feature_names(), tfidf_vectorizer.idf_))

    print("BUILDING WORD VECS WITH TFIDF AND W2V>>>>")
    train_x = labelize(x_train, "TRAIN")
    test_x = labelize(x_test, "TEST")

    train_vecs_w2v = np.concatenate([sentence_vectorizer(z, w2v, tfidf, 200) for z in map(lambda x: x, train_x)])
    train_vecs_w2v = scale(train_vecs_w2v)

    test_vecs_w2v = np.concatenate([sentence_vectorizer(z, w2v, tfidf, 200) for z in map(lambda x: x, test_x)])
    test_vecs_w2v = scale(test_vecs_w2v)

    print("VECOTRIZING LABELS>>>>")
    y_train = train_data[3].as_matrix()
    y_test = test_data[3].as_matrix()

    y_tokenizer = Tokenizer()
    y_tokenizer.fit_on_texts(y_train)
    y_train = index_text(y_train, y_tokenizer)
    y_test = index_text(y_test, y_tokenizer)


    print("FITTING FNN>>>>")

    # FNN
    model = model(train_vecs_w2v.shape[1], 4)
    model.fit(
        x=train_vecs_w2v,
        y=y_train,
        batch_size=64,
        epochs=20,
        verbose=2,
        validation_split=0.1)

    print("VALIDATING ON TEST SET>>>>")
    score = model.evaluate(test_vecs_w2v, y_test, batch_size=64, verbose=2)

    print('\nTEST SCORE:', score[0])
    print('TEST ACCURACY:', score[1])

    """
    Epoch 1/20
    2s - loss: 0.9226 - acc: 0.6889 - val_loss: 0.8118 - val_acc: 0.6863
    Epoch 2/20
    2s - loss: 0.7925 - acc: 0.6938 - val_loss: 0.7904 - val_acc: 0.6863
    Epoch 3/20
    2s - loss: 0.7823 - acc: 0.6938 - val_loss: 0.7834 - val_acc: 0.6863
    Epoch 4/20
    2s - loss: 0.7773 - acc: 0.6938 - val_loss: 0.7802 - val_acc: 0.6863
    Epoch 5/20
    2s - loss: 0.7755 - acc: 0.6938 - val_loss: 0.7784 - val_acc: 0.6863
    Epoch 6/20
    2s - loss: 0.7739 - acc: 0.6938 - val_loss: 0.7765 - val_acc: 0.6863
    Epoch 7/20
    2s - loss: 0.7721 - acc: 0.6938 - val_loss: 0.7755 - val_acc: 0.6863
    Epoch 8/20
    2s - loss: 0.7715 - acc: 0.6938 - val_loss: 0.7752 - val_acc: 0.6863
    Epoch 9/20
    1s - loss: 0.7711 - acc: 0.6938 - val_loss: 0.7746 - val_acc: 0.6863
    Epoch 10/20
    1s - loss: 0.7707 - acc: 0.6938 - val_loss: 0.7738 - val_acc: 0.6863
    Epoch 11/20
    2s - loss: 0.7708 - acc: 0.6938 - val_loss: 0.7737 - val_acc: 0.6863
    Epoch 12/20
    1s - loss: 0.7699 - acc: 0.6938 - val_loss: 0.7735 - val_acc: 0.6863
    Epoch 13/20
    1s - loss: 0.7706 - acc: 0.6938 - val_loss: 0.7731 - val_acc: 0.6863
    Epoch 14/20
    1s - loss: 0.7691 - acc: 0.6938 - val_loss: 0.7729 - val_acc: 0.6863
    Epoch 15/20
    2s - loss: 0.7698 - acc: 0.6938 - val_loss: 0.7729 - val_acc: 0.6863
    Epoch 16/20
    2s - loss: 0.7702 - acc: 0.6938 - val_loss: 0.7729 - val_acc: 0.6863
    Epoch 17/20
    1s - loss: 0.7695 - acc: 0.6938 - val_loss: 0.7729 - val_acc: 0.6863
    Epoch 18/20
    2s - loss: 0.7703 - acc: 0.6938 - val_loss: 0.7726 - val_acc: 0.6863
    Epoch 19/20
    2s - loss: 0.7691 - acc: 0.6938 - val_loss: 0.7727 - val_acc: 0.6863
    Epoch 20/20
    2s - loss: 0.7687 - acc: 0.6938 - val_loss: 0.7722 - val_acc: 0.6863
    VALIDATING ON TEST SET>>>>
    
    TEST SCORE: 0.779745739508
    TEST ACCURACY: 0.655105222182
    """