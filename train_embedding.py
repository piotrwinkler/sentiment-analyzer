import string
import gensim
import csv
import argparse
from prepare_data import prepare
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def train(data_dir, review_dir, embedding_dir, model_dir):
    """
    This module uses natural language toolkit (nltk) to divide reviews into single words. Based on that gensim model is
    trained in order to provide embedding vectors.
    """
    embedding_dim = 100

    x_train, x_test, _, _ = prepare(data_dir)
    all_reviews = x_test + x_train
    review_lines = []
    counter = 0
    for line in all_reviews:
        tokens = word_tokenize(line)
        tokens = [w.lower() for w in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
        review_lines.append(words)
        counter += 1
        if counter % 10000 == 0:
            print(counter, '/', len(all_reviews))

    with open(review_dir, 'w') as f:
        wr = csv.writer(f)
        wr.writerows(review_lines)

    print(review_lines[0])
    print(review_lines[3])
    print(len(review_lines))

    model = gensim.models.Word2Vec(sentences=review_lines, size=embedding_dim, window=5, workers=4, min_count=10)

    words = list(model.wv.vocab)
    print('Vocabulary size: %d' % len(words))

    model_dir = model_dir
    model.save(model_dir)

    filename = embedding_dir
    model.wv.save_word2vec_format(filename, binary=False)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', '-d', help='path to input/source files',
                        default='yelp_test_100k_negative.csv yelp_negative_300k.txt yelp_positive_300k.txt '
                                'yelp_test_100k_positive.txt')

    parser.add_argument('--review_dir', '-r', help='save reviews data here', default='review_lines.csv')

    parser.add_argument('--embedding_dir', '-e', help='save embedding here', default='word2vec.txt')

    parser.add_argument('--model_dir', '-m', help='save gensim model here', default='gensim.model')

    return parser.parse_args()


def main():
    args = parse_args()

    train(args.data_dir, args.review_dir, args.embedding_dir, args.model_dir)


if __name__ == "__main__":
    main()
