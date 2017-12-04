from markovify import Text, Chain, NewlineText
from markovify.chain import BEGIN, END
from nltk.corpus import movie_reviews, gutenberg
from nltk.tokenize import word_tokenize
import random
import pronouncing

DEFAULT_MAX_OVERLAP_RATIO = 0.7
DEFAULT_MAX_OVERLAP_TOTAL = 15
DEFAULT_TRIES = 10


def tokens(sent):
    if not isinstance(sent, str):
        return []
    tokens = word_tokenize(sent)
    symbols = """!@#$%^&*()_+-=[]{}:"'\|?/>.<,`~"""
    tokens = [token.strip(symbols).lower() for token in tokens]
    tokens = [token for token in tokens if token]

    return tokens


def word_sounds(word):
    transcription = pronouncing.phones_for_word(word)
    counter = 0
    #     print ("Transcript", transcription)
    for word in transcription:
        for sound in word:
            for char in sound:
                if char.isdigit():
                    counter += 1
    return counter


def sentence_sounds(sent):
    return sum([word_sounds(token) for token in tokens(sent)])


def reverse(s):
    return ' '.join(s.split(' ')[::-1])


class Rhymer(NewlineText):
    def __init__(self, input_text, state_size=2, chain=None, parsed_sentences=None, retain_original=True):
        corpus = input_text
        for c in '.?!':
            corpus = corpus.replace(c, '\n')

        corpus = reverse(corpus).lower()

        super(NewlineText, self).__init__(corpus, state_size, chain, parsed_sentences, retain_original)

    def in_vocab(self, word):
        return word in self.chain.model

    def make_sentence_with_end(self, beginning, **kwargs):
        tries = kwargs.get('tries', DEFAULT_TRIES)
        mor = kwargs.get('max_overlap_ratio', DEFAULT_MAX_OVERLAP_RATIO)
        mot = kwargs.get('max_overlap_total', DEFAULT_MAX_OVERLAP_TOTAL)
        test_output = kwargs.get('test_output', True)
        max_words = kwargs.get('max_words', None)

        if len(beginning.split(' ')) > 1:
            print('only one word as beginning is expected)')
            return None

        init_state = (BEGIN,) * (self.state_size - 1) + (beginning,)
        if not self.in_vocab(init_state):
            return None

        output = None
        prefix = beginning
        max_tries = tries
        tr = 0
        while not output and tr < max_tries:
            words = ([prefix] + self.chain.walk(init_state))[::-1]
            tr += 1
            if self.test_sentence_output(words, mor, mot):
                output = self.word_join(words)

        return output

    def make_rhyme(self, sentence):
        target_sounds_count = sentence_sounds(sentence)
        last_word = tokens(sentence)[-1]
        target_rhymes = pronouncing.rhymes(last_word)
        target_sent = None

        tries = 0
        while not target_sent:
            random.shuffle(target_rhymes)
            for rhyme in target_rhymes:
                sent = self.make_sentence_with_end(rhyme)
                tries += 1

                if tries > 1000:
                    target_sent = 'Too difficult. Sorry, try again later :('
                    break

                if sent and sentence_sounds(sent) == target_sounds_count:
                    target_sent = sent
                    break

        target_sent = target_sent[0].upper() + target_sent[1:]
        return target_sent

# rhymer = Rhymer(movie_reviews.raw(), state_size=3)