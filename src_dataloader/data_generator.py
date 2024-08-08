import os
import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, preprocessed_path, step, batch_size=32, shuffle=True, file_list=None, **kwargs):
        super().__init__(**kwargs)
        self.preprocessed_path = preprocessed_path
        self.step = step
        self.batch_size = batch_size
        self.shuffle = shuffle
        if file_list is not None:
            self.data_files = file_list
        else:
            self.data_files = [f for f in os.listdir(preprocessed_path) if f.endswith('.npy')]
        
        self.indexes = np.arange(len(self.data_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.data_files) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = [self._load_data(self.data_files[i]) for i in batch_indexes]
        
        batch_data = [data for data in batch_data if data is not None]

        if not batch_data:
            return np.array([]), np.array([])
        
        batch_x, batch_y = zip(*batch_data)

        batch_x = tuple(tf.convert_to_tensor(item) for item in zip(*batch_x))
        batch_y = tf.convert_to_tensor(batch_y)
        
        return batch_x, batch_y

    def _load_data(self, filename):
        file_path = os.path.join(self.preprocessed_path, filename)
        sample = np.load(file_path, allow_pickle=True).item()

        X = (sample['photometry'], sample['metadata'].to_numpy(), sample['images'])
        y = sample[self.step]

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
