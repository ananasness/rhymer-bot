import config
import telebot
from rhymer import Rhymer
from rhymer_nn import RhymerNN

data_samples_count = 20000

"""pip install -r requirements.txt"""

bot = telebot.TeleBot(config.bot_token)
rhymer_hmm = None
rhymer_word = None
chat_model = {}


hello_message = (
""" Hello, I'm rhymer-bot. \n
Please choose the way to make rhyme.\n
/hmm - Hidden Markov model\n
/word_rnn - Word-based LSTM neural network\n""")
#/char_rnn - Char-based LSTM neural network\n""")

start_message = "Let's make some poem!"


@bot.message_handler(commands=["start"])
def start(message):
    chat_model[message.chat.id] = None
    bot.send_message(message.chat.id, hello_message)


@bot.message_handler(commands=["hmm"])
def repeat_all_messages(message):
    chat_model[message.chat.id] = rhymer_hmm
    bot.send_message(message.chat.id, start_message)


@bot.message_handler(commands=["word_rnn"])
def repeat_all_messages(message):
    chat_model[message.chat.id] = rhymer_word
    bot.send_message(message.chat.id, start_message)


@bot.message_handler(content_types=["text"])
def answer(message):
    if chat_model.get(message.chat.id):
        sent = message.text
        bot.send_message(message.chat.id, chat_model[message.chat.id].make_rhyme(sent))
    else:
        start(message)


def dataset_2():
    with open('../data/chat_bot_data_processed(encoded).txt', 'r', encoding='utf-8') as data_file:
        return data_file.read()
    return dataset_1()


def dataset_1():
    from nltk.corpus import movie_reviews
    return movie_reviews.raw()


def dataset_3():
    with open('../data/songs_data_processed_2.txt', 'r', encoding='utf-8') as data_file:
        return data_file.read()[:data_samples_count]

if __name__ == '__main__':

    rhymer_hmm = Rhymer(dataset_3(), state_size=3, retain_original=True)
    rhymer_word = RhymerNN('word_model')

    print('Data processing is finished. Bot is ready.')

    bot.polling(none_stop=True)

