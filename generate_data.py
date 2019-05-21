import numpy as np
from keras.utils import Sequence
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


class Generator(Sequence):
    """
    This generator prepares a batches of data during learning process in order to limit the memory needed to store the
    data.
    """
    def __init__(self, batch_size, x_set, y_set, tokenizer_obj, max_lenght):
        self.x_set, self.y_set = x_set, y_set
        self.batch_size = batch_size
        self.tokenizer_obj = tokenizer_obj
        self.max_lenght = max_lenght

    def __len__(self):
        return int(np.ceil(len(self.x_set) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x_set[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x_tokens = self.tokenizer_obj.texts_to_sequences(batch_x)
        batch_x_pad = pad_sequences(batch_x_tokens, maxlen=self.max_lenght, padding='post')
        batch_y = self.y_set[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x_pad, batch_y
