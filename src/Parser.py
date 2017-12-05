import csv
import string
import pickle
import pandas as pd

#parse movie dialogs data
def movie_data_parse():
    with open("..\data\movie_lines.tsv") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        data = [conv[4:] for conv in tsvreader]
#remove empty lines
    for sentence in data:
        if not len(sentence):
            data.remove(sentence)

#remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    data_punct_removed = ""
    sentence_without_punct = ""

    for sentence in data:
        tuple_without_punct = str(sentence).translate(translator)
        ''.join(tuple_without_punct)
        for char in tuple_without_punct:
            sentence_without_punct += str(char)
        sentence_without_punct += "\n"

#make a dump of the data
    pickle.dump(sentence_without_punct, open("data.p", "wb"))


def songs_data():
    #read data into pandas dictionary
    df = pd.read_csv('songdata.csv')
    saved_column = df['text']

    #write all data into a file
    with open("songs_data_processed_2.txt", "a") as myfile:
        for column in saved_column:
            myfile.write(column)

