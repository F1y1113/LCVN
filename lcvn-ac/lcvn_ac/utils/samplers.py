import math
import numpy as np
from torch.utils.data import Sampler


class EquiSampler(Sampler):
    """Equidistant batch sampler.

    Yields n (where n==batch_size) equidistant indices, steps through the dataset by adding the sequence length to each index and yielding the new set of indices.
    """

    def __init__(self, data_size, seq_len, batch_size, init_idx=None):
        self.data_size = data_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.init_idx = init_idx
        self.chunk_size = math.ceil(self.data_size / self.batch_size)
        self.n_steps = math.ceil(self.chunk_size / self.seq_len)
        print("Chunk size:", self.chunk_size)
        print("n steps:", self.n_steps)

    def __iter__(self):
        if self.init_idx is None:
            init_idx = np.random.randint(self.data_size)
        else:
            init_idx = self.init_idx
        for i in range(self.n_steps):
            iters = []
            for j in range(self.batch_size):
                print(f"[DEBUG EquiSampler] Current self.data_size = {self.data_size}")
                start_idx = (init_idx + i * self.seq_len + j * self.chunk_size) % self.data_size
                iters.append(start_idx)
            yield iters

    def __len__(self):
        return self.n_steps
