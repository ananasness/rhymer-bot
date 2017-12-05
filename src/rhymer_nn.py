import random
import datetime
from train_word_rnn import WordRNN, BEGIN, END
import numpy as np
from pronouncing import rhymes

random.seed(datetime.datetime.now())
class RhymerNN():
    def __init__(self, dir_path):
        print('RNN model loading start')
        self.model, self.token2id, self.id2token = WordRNN.load_trained_model(dir_path)
        print('RNN model loading finish')

    def make_rhyme(self, sent):
        last_word = sent.split(' ')[-1].lower()

        rhms = rhymes(last_word)
        random.shuffle(rhms)

        for rhyme in rhms:
            print(rhyme)
            if self.token2id.get(rhyme):
                gen_sent = self.gen_words_with_last(rhyme)
                return gen_sent

        else:
            return 'Sorry. I do not have this word in my vocabulary :('

    def gen_words_with_last(self, start=None):
        begin = [] if not start else start.lower().split(' ')
        sent = [END] + begin
        new_word = None

        j = 0
        while new_word != BEGIN and j < 20:
            j += 1
            input_words = [self.token2id[token] for token in sent]

            ids_sent = np.array([input_words])
            i = np.argmax(self.model.predict(ids_sent)[0])
    #         print(ids_sent)
            new_word = self.id2token[i]
            sent.append(new_word)

        sent = sent[1:-1]
        return ' '.join(sent[::-1])


