import numpy as np
import utils
import os

class Buffer(object):
    def __init__(self, buffer_size, batch_size, image_dims=[256,256,3], random_seed=0):
        self.rng = np.random.RandomState(random_seed)
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.idx = 0
        self.data = np.zeros([self.buffer_size] + image_dims) # For both domain
        self.label = np.zeros([self.buffer_size])

    def push(self, batches, labels = np.array([None])):
        batch_size = len(batches)
        if self.idx + batch_size > self.buffer_size:
            random_idx1 = self.rng.choice(self.idx, batch_size)
            self.data[random_idx1] = batches
            if labels.all() != None:
                self.label[random_idx1] = labels
            '''
            random_idx1 = self.rng.choice(self.idx, self.batch_size)#/2)
            random_idx2 = self.rng.choice(batch_size, self.batch_size)#/2)
            self.data[random_idx1] = batches[random_idx2]
            '''
        else:
            self.data[self.idx:self.idx+batch_size] = batches
            if labels.all() != None:
                self.label[self.idx:self.idx+batch_size] = labels
            self.idx += batch_size

    def sample(self, n=None):
        if n is None:
            n = self.batch_size
        assert self.idx >= n, "not enough data is pushed"
        random_idx = self.rng.choice(self.idx, n)
        return self.data[random_idx]

    def print_all(self, print_dir, dimA=3, dimB=3):
        if not os.path.exists(print_dir):
            os.mkdir(print_dir)
        imgA, imgB = np.split(self.data, 2, axis=3)

        names=[]
        for i in range(self.idx):
            names.append(str(i))
        
        utils.save_examples(imgA, print_dir+'/imgA', name=names, num=self.idx, label=self.label)
        utils.save_examples(imgB, print_dir+'/imgB', name=names, num=self.idx, label=self.label)

        return

