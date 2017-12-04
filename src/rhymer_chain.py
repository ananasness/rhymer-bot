from markovify import Chain

BEGIN = "___BEGIN__"
END = "___END__"
MAX_LEN = 5

class RhymerChain(Chain):

    def gen(self, init_state=None):

        word_count = 1 if init_state else 0
        state = init_state or (BEGIN,) * self.state_size
        while word_count <= MAX_LEN:
            next_word = self.move(state)
            word_count += 1
            if next_word == END: break
            yield next_word
            state = tuple(state[1:]) + (next_word,)

        else:
            return None
