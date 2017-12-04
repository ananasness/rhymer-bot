import config
import telebot
from rhymer import Rhymer

bot = telebot.TeleBot(config.bot_token)
rhymer = None



@bot.message_handler(commands=["start"])
def repeat_all_messages(message):
    bot.send_message(message.chat.id, "Hello, I'm rhymer bot. Let's make some poem!")


@bot.message_handler(content_types=["text"])
def answer(message):
    sent = message.text
    bot.send_message(message.chat.id, rhymer.make_rhyme(sent))



if __name__ == '__main__':
    with open('../data/chat_bot_data_processed(encoded).txt', 'r', encoding='utf-8') as data_file:
        rhymer = Rhymer(data_file.read(), state_size=3)
        print('Data processing is finished')

    bot.polling(none_stop=True)

