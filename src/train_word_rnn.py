import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.utils import to_categorical
import numpy as np
import pickle
from nltk.corpus import movie_reviews
from nltk import word_tokenize, sent_tokenize

BEGIN = '<begin>'
END = '<end>'
full = {"n't": 'not',
        "'s": 'is',
        "'ll": 'will',
        "'ve": 'have',
        "'d": 'had',
        "'re": 'are',
        "'m": 'am',
        "wo": 'will'}

DATA_SIZE = 150000


class WordRNN:

    def __init__(self):
        self.model = None
        self.data_size = 150000
        self.input_len = 5
        self.epochs = 20
        self.vocab = []
        self.vocab_len = 0
        self.token2id = {}
        self.id2token = {}
        self.X_train = []
        self.Y_train = []

    @classmethod
    def get_model(cls, vocab_len):
        model = Sequential()
        model.add(Embedding(input_dim=vocab_len, output_dim=500))
        model.add(LSTM(128, return_sequences=False, implementation=1))
        model.add(Dropout(0.25))
        # model.add(LSTM(128, implementation=1))
        # model.add(Dropout(0.))
        model.add(Dense(vocab_len, activation='softmax'))

        model.compile(optimizer=keras.optimizers.rmsprop(lr=0.007), loss='categorical_crossentropy')

        return model

    def data_preprocessed(self):

        syms = '[]()"'
        data = movie_reviews.raw().lower()
        for sym in list(syms) + ['...']:
            data = data.replace(sym, '')

        corpus = data[:self.data_size]
        self.vocab = set(self.tokens_filter(corpus) + [BEGIN, END])
        self.vocab_len = len(self.vocab) + 1
        self.token2id = {t: i for i, t in enumerate(self.vocab, start=1)}
        self.id2token = {i: t for i, t in enumerate(self.vocab, start=1)}

        return corpus

    def train_data_prepared(self):
        train_set_x = []
        train_set_y = []

        for sent in sent_tokenize(self.data_preprocessed()):

            ids = self.sent2ids(str(sent))
            #     print(sent, [id2token[i] for i in ids])

            #     print('\r{}/{}'.format(i, len()), end='')
            if ids is None:
                continue

            ids = ids[::-1]
            for i in range(len(ids) - self.input_len):
                train_set_x.append(ids[i: i + self.input_len])
                train_set_y.append(to_categorical(ids[i + self.input_len], self.vocab_len))

        self.X_train = np.array(train_set_x, dtype=np.int32)
        self.Y_train = np.array(train_set_y)[:, 0, :]

        return self.X_train, self.Y_train

    def train(self):
        self.model = self.get_model(self.vocab_len)
        self.model.fit(*self.train_data_prepared(), batch_size=64, epochs=self.epochs)

    def tokens_filter(self, sent):

        tokens = word_tokenize(sent)
        symbols = """!@#$%^&*()_+-=[]{}:"'\|?/>.<,`~1234567890"""
        tokens = [self.fullform(token.lower()).strip(symbols) for token in tokens]
        tokens = [token for token in tokens if token]

        return tokens

    def fullform(self, token):
        ft = full.get(token, None)

        return ft or token

    def sent2ids(self, sent):
        tokens = self.tokens_filter(sent)
        if len(tokens) == 0:
            return None
        tokens = [BEGIN] + tokens + [END]

        return [self.token2id[token] for token in tokens]

    def words2input(self, words):
        ids = [self.token2id[word] for word in words]
        return np.array(ids)

    def save_model(self):
        self.model.save('model/model.h5')
        self.model.save_weights('model/weights.hdf5')
        json.dump(self.token2id, open('model/token2id.json', 'w'))
        json.dump(self.id2token, open('model/id2token.json', 'w'))

    @classmethod
    def load_trained_model(cls, path):

        t2i = pickle.load(open(path + '/token2id.p', 'rb'))
        i2t = pickle.load(open(path + '/id2token.p', 'rb'))
        model = cls.get_model(len(t2i) + 1)

        model.load_weights(path+'/weights.h5')

        return model, t2i, i2t
