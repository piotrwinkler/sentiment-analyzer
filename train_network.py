import csv
import random
import argparse
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import to_categorical
from keras import optimizers
from prepare_data import prepare
from generate_data import Generator
from use_preembedding import use
from time import time


def train_network(data_dir, review_dir, embedding_dir, models_dir, logs_dir, batch,
                  epochs, transfer):
    """
    In this function structure of neural network is defined. All the training takes place here based on provided data.
    """
    embedding_dim = 100

    _, _, y_train, y_test = prepare(data_dir)
    embedding_matrix, tokenizer_obj, num_words = use(embedding_dir)

    with open(review_dir, 'r') as f:
        reader = csv.reader(f)
        review_lines = list(list(rec) for rec in csv.reader(f, delimiter=','))

    avg_lenght = 0
    for i in review_lines:
        avg_lenght += len(i)
    avg_lenght = int(avg_lenght / len(review_lines) + 100)

    print('Data prepared')
    print('')

    model = Sequential()
    if transfer:
        embedding_layer = Embedding(num_words, embedding_dim,
                                    input_length=avg_lenght,
                                    trainable=False)
        embedding_layer.build((None,))
        embedding_layer.set_weights([embedding_matrix])
    else:
        embedding_layer = Embedding(num_words, embedding_dim,
                                    input_length=avg_lenght,
                                    trainable=True)
    model.add(embedding_layer)
    model.add(Flatten())
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    adam = optimizers.adam(lr=0.001)
    model.compile(loss="binary_crossentropy",
                  optimizer=adam,
                  metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath=models_dir,
                                   verbose=1, save_best_only=False, save_weights_only=False, period=1)
    tensorboard = TensorBoard(log_dir=logs_dir.format(time()))

    x_train = review_lines[200000:]
    x_test = review_lines[:200000]

    x_shuffle_train = list(zip(x_train, y_train))
    random.shuffle(x_shuffle_train)
    x_train, y_train = zip(*x_shuffle_train)
    x_train = list(x_train)
    y_train = list(y_train)

    x_shuffle_test = list(zip(x_test, y_test))
    random.shuffle(x_shuffle_test)
    x_test, y_test = zip(*x_shuffle_test)
    x_test = list(x_test)
    y_test = list(y_test)

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    training_batch_generator = Generator(batch, x_train, y_train, tokenizer_obj, avg_lenght)
    validation_batch_generator = Generator(batch, x_test, y_test, tokenizer_obj, avg_lenght)

    print('Model prepared, start training...')
    model.fit_generator(generator=training_batch_generator,
                        steps_per_epoch=(300000 // batch),
                        epochs=epochs,
                        verbose=1,
                        validation_data=validation_batch_generator,
                        validation_steps=(200000 // batch),
                        use_multiprocessing=False,
                        max_queue_size=1,
                        callbacks=[checkpointer, tensorboard])

    if transfer:
        model.layers[0].trainable = True
        sgd = optimizers.sgd(lr=0.00001)
        model.compile(loss="binary_crossentropy",
                      optimizer=sgd,
                      metrics=['accuracy'])
        model.fit_generator(generator=training_batch_generator,
                            steps_per_epoch=(300000 // batch),
                            epochs=2,
                            verbose=1,
                            validation_data=validation_batch_generator,
                            validation_steps=(200000 // batch),
                            use_multiprocessing=False,
                            max_queue_size=1,
                            callbacks=[checkpointer, tensorboard])


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', '-d', help='path to input/source files',
                        default='yelp_test_100k_negative.csv yelp_negative_300k.txt yelp_positive_300k.txt '
                                'yelp_test_100k_positive.txt')

    parser.add_argument('--review_dir', '-r', help='directory of reviews csv file', default='review_lines.csv')

    parser.add_argument('--embedding_dir', '-g', help='directory of pretrained embedding file', default='word2vec.txt')

    parser.add_argument('--models_dir', '-m', help='specifies saving directory for models',
                        default='model.{epoch:02d}.hdf5')

    parser.add_argument('--logs_dir', '-l', help='specifies saving directory for logs', default='logs')

    parser.add_argument('--batch_size', '-b', help='number of examples in one batch', type=int, default=128)

    parser.add_argument('--epochs', '-e', help='number of training epochs', type=int, default=2)

    parser.add_argument('--transfer', '-t', help='flag for transfer learning', type=bool, default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    train_network(args.data_dir, args.review_dir, args.embedding_dir, args.models_dir, args.logs_dir, args.batch_size,
                  args.epochs, args.transfer)


if __name__ == "__main__":
    main()
