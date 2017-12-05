from markovify import NewlineText
from markovify.chain import BEGIN, END
from nltk.tokenize import word_tokenize
import nltk
import random
from datetime import datetime
import pronouncing
from rhymer_chain import RhymerChain
import re

DEFAULT_MAX_OVERLAP_RATIO = 0.7
DEFAULT_MAX_OVERLAP_TOTAL = 15
DEFAULT_TRIES = 10


random.seed(datetime.now())
#tokenization
def tokens(sent):
    if not isinstance(sent, str):
        return []
    tokens = word_tokenize(sent)
    symbols = """!@#$%^&*()_+-=[]{}:"'\|?/>.<,`~"""
    tokens = [token.strip(symbols).lower() for token in tokens]
    tokens = [token for token in tokens if token]

    return tokens

#count amount of vowel sounds is a word
#Example: "salary" has 3 sounds
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

#amount of all sounds in a sentence
def sentence_sounds(sent):
    return sum([word_sounds(token) for token in tokens(sent)])

#reverse input string
def reverse(s):
    return ' '.join(s.split(' ')[::-1])


class Rhymer(NewlineText):

    def __init__(self, input_text, state_size=2, chain=None, parsed_sentences=None, retain_original=True):
        #data preprocessing
        corpus = input_text.lower()
        self.vocab = set(input_text.split(' '))
        print('Number of words in the vocabulary is {}.'. format(len(self.vocab)))
        for c in '.?!':
            corpus = corpus.replace(c, '\n')

        input_text = reverse(corpus)

        can_make_sentences = parsed_sentences is not None or input_text is not None
        self.retain_original = retain_original and can_make_sentences
        self.state_size = state_size

        #create a model
        if self.retain_original:
            self.parsed_sentences = parsed_sentences or list(self.generate_corpus(input_text))

            # Rejoined text lets us assess the novelty of generated sentences
            self.rejoined_text = self.sentence_join(map(self.word_join, self.parsed_sentences))
            self.chain = chain or RhymerChain(self.parsed_sentences, state_size)
        else:
            if not chain:
                parsed = parsed_sentences or self.generate_corpus(input_text)
            self.chain = chain or RhymerChain(parsed, state_size)

        print('The model is ready.')


    #check whether given word is in a vocabulary of beginning words
    def in_vocab(self, word):
        if len(word) > 1:
            init_state = (BEGIN,) * (self.state_size - 1) + (word,)
            return init_state in self.chain.model

        return False

    #returns sentences which end with a given word
    def make_sentence_with_end(self, beginning, **kwargs):
        tries = kwargs.get('tries', DEFAULT_TRIES)
        mor = kwargs.get('max_overlap_ratio', DEFAULT_MAX_OVERLAP_RATIO)
        mot = kwargs.get('max_overlap_total', DEFAULT_MAX_OVERLAP_TOTAL)
        test_output = kwargs.get('test_output', True)
        max_words = kwargs.get('max_words', None)

        if len(beginning.split(' ')) > 1:
            print('only one word as beginning is expected)')
            return None

        #initialize the first state
        init_state = (BEGIN,) * (self.state_size - 1) + (beginning,)

        output = None
        prefix = beginning
        max_tries = tries
        tr = 0
        #try to generate sentences till we have the right one
        #there is a fixed number of trials to generate a sentence
        while not output and tr < max_tries:
            gen_words = self.chain.walk(init_state)
            tr += 1
            if not gen_words: continue

            words = ([prefix] + gen_words)[::-1]

            if self.test_sentence_output(words, mor, mot):
                output = self.word_join(words)

        return output

    # returns list of similar words
    def signature(self, word):
        # mappings are done to find similar words is they were misspelled

        mappings = [('ph', 'f'), ('ght', 't'), ('^kn', 'n'), ('qu', 'kw'),
                    ('[aeiou]+', 'a'), (r'(.)\1', r'\1')]
        for patt, repl in mappings:
            word = re.sub(patt, repl, word)
        pieces = re.findall('[^aeiou]+', word)
        return ''.join(char for piece in pieces for char in sorted(piece))[:8]


    # rank similar words by similarity distance (which word was more likely to appear)
    def rank(self, word, wordlist):
        ranked = sorted((nltk.edit_distance(word, w), w) for w in wordlist)
        return [word for (_, word) in ranked]


    #return list of words which make rhyme with a given word
    #level specifies number of sounds which should be similar (n-last should sound the same)
    def rhyme(self, inp, level):
        entries = nltk.corpus.cmudict.entries()
        syllables = [(word, syl) for word, syl in entries if word == inp]
        rhymes = []
        for (word, syllable) in syllables:
            rhymes += [word for word, pron in entries if pron[-level:] == syllable[-level:]]
        return set(rhymes)

    def make_rhyme(self, sentence):
        target_sounds_count = sentence_sounds(sentence)
        last_word = tokens(sentence)[-1]
        target_rhymes = pronouncing.rhymes(last_word)
        target_rhymes.extend(self.rhyme(last_word, 1))
        target_rhymes = [rhyme for rhyme in target_rhymes if self.in_vocab(rhyme)]
        target_sent = None

        print("Processing word: {}, target rhymes: {}, target sounds num: {}".format(last_word,
                                                                                     len(target_rhymes),
                                                                                     target_sounds_count))
        if not target_rhymes: target_sent = 'Too difficult last word. ' \
                                          'Sorry, try again later :('

        tries = 0
        possible_sent = []
        while not target_sent:

            random.shuffle(target_rhymes)
            for rhyme in target_rhymes:
                gen_sent = self.make_sentence_with_end(rhyme)
                tries += 1
                if tries > 100:
                    if possible_sent:
                        target_sent = random.choice(possible_sent)
                        print('Model cannot generate suitable rhyme')
                        break
                    else:
                        tries = 0

                # print(sentence_sounds(sent), end=',')
                # print(gen_sent)
                if gen_sent != None:
                    possible_sent.append(gen_sent)
                    print('{}. {}'.format(tries, possible_sent[-1]))
                    if False: #sentence_sounds(gen_sent) == target_sounds_count:
                        target_sent = gen_sent
                        break

        return self.postproc(target_sent)

    def postproc(self, sent):
        print(sent)

        while not sent[0].isalpha():
            sent = sent[1:]

        sent = sent[0].upper() + sent[1:]
        sent.replace(' ,', ',')
        return sent
