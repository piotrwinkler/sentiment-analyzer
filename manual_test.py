import gensim
import csv
import argparse
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


def test_gensim_model(directory, word):
    """
    This module allows to test provided gensim model by illustrating similarity between words.
    """
    model = gensim.models.Word2Vec.load(directory)

    print('Similar to:', word)
    print(model.wv.most_similar(word))
    print('')


def test_nn_model(review_dir, embedding_dir, nn_dir, sentence_nr):
    """
    This module allows to test NN model by predicting sentiment on provided review.
    """
    with open(review_dir, 'r') as f:
        reader = csv.reader(f)
        review_lines = list(list(rec) for rec in csv.reader(f, delimiter=','))

    avg_lenght = 0
    for i in review_lines:
        avg_lenght += len(i)
    avg_lenght = int(avg_lenght/len(review_lines) + 100)

    words = []
    f = open(embedding_dir, encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        words.append(word)
    f.close()

    words = words[1:]

    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(words)

    sentence = [review_lines[sentence_nr]]
    print(sentence)
    print('')

    sentence_token = tokenizer_obj.texts_to_sequences(sentence)
    print(sentence_token)
    print('')

    sentence_pad = pad_sequences(sentence_token, maxlen=avg_lenght, padding='post')

    model = load_model(nn_dir)
    prediction = model.predict(sentence_pad)
    print(prediction)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gensim_dir', '-g', help='directory of trained gensim model',
                        default='gensim.model')

    parser.add_argument('--review_dir', '-r', help='directory of reviews csv file', default='review_lines.csv')

    parser.add_argument('--embedding_dir', '-e', help='directory of pretrained embedding file', default='word2vec.txt')

    parser.add_argument('--nn_dir', '-n', help='directory of trained nn model', default='model.02.hdf5')

    parser.add_argument('--test_nr', '-t', help='1-gensim, 2-nn, 3-both', type=int, default=3)

    parser.add_argument('--word', '-w', help='test similarity on this word', default='good')

    parser.add_argument('--sentence_nr', '-s', help='number of review for neural network', type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.test_nr == 3 or args.test_nr == 1:
        test_gensim_model(args.gensim_dir, args.word)
    if args.test_nr == 3 or args.test_nr == 2:
        test_nn_model(args.review_dir, args.embedding_dir, args.nn_dir, args.sentence_nr)


if __name__ == "__main__":
    main()
