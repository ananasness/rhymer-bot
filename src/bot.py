import config
import telebot
from rhymer import Rhymer

"""pip install -r requirements.txt"""

bot = telebot.TeleBot(config.bot_token)
rhymer = None


@bot.message_handler(commands=["start"])
def repeat_all_messages(message):
    bot.send_message(message.chat.id, "Hello, I'm rhymer bot. Let's make some poem!")


@bot.message_handler(content_types=["text"])
def answer(message):
    sent = message.text
    bot.send_message(message.chat.id, rhymer.make_rhyme(sent))


def dataset_2():
    with open('../data/chat_bot_data_processed(encoded).txt', 'r', encoding='utf-8') as data_file:
        return data_file.read()

    return dataset_1()


def dataset_1():
    from nltk.corpus import movie_reviews
    return movie_reviews.raw()

def dataset_3():
    with open('../data/songs_data_processed_2.txt', 'r', encoding='utf-8') as data_file:
        return data_file.read()[0:20000]

if __name__ == '__main__':

    rhymer = Rhymer(dataset_3(), state_size=3, retain_original=True)
    print('Data processing is finished')

    bot.polling(none_stop=True)

