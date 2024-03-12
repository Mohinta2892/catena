import gunpowder as gp
import numpy as np


class Unsqueeze(gp.BatchFilter):

    def __init__(self, array):
        self.array = array

    def prepare(self, request):
        pass

    def process(self, batch, request):
        # always add a dim in front of the array
        # print(f"BEFORE UNSQUEEZING---->>>>> {batch[self.array].data.shape}")
        batch[self.array].data = np.expand_dims(batch[self.array].data, axis=0)
        # print(f"AFTER UNSQUEEZING---->>>>> {batch[self.array].data.shape}")


class Squeeze(gp.BatchFilter):

    def __init__(self, array):
        self.array = array

    def prepare(self, request):
        pass

    def process(self, batch, request):
        # always add a dim in front of the array
        # print(f"BEFORE UNSQUEEZING---->>>>> {batch[self.array].data.shape}")
        batch[self.array].data = np.squeeze(batch[self.array].data, axis=0)
        # print(f"AFTER UNSQUEEZING---->>>>> {batch[self.array].data.shape}")
