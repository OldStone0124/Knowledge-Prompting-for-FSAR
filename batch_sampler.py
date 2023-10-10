import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        # self.m_ind is the sample indices for all samples belong to the ith class

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            if self.n_cls == 64:
                classes = torch.arange(self.n_cls)
            else:
                classes = torch.randperm(len(self.m_ind))[:self.n_cls]
                # random permute the order of classes, the select n_cls classes from it

            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                # random permute the order of the samples of a certain class

                batch.append(l[pos])


            batch = torch.stack(batch).t().reshape(-1)

            yield batch
