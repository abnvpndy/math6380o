import os
from time import time

import torch
from torch.utils.data.dataset import TensorDataset

"""generic class to extract features from pretrained nets"""


class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def features(self, dataloader, save_to_disk=True, train=True):
        feat_coll = []
        label_coll = []
        for batch_id, [features, labels] in enumerate(dataloader):
            # sample is a list with the first element corresponding to the images
            print("Batch {}, features shape: {}, labels shape: {}".format(batch_id, features.shape, labels.shape))
            features = features.to(self.device)
            labels = labels.to(self.device)
            t1 = time()
            out = self.model(features)
            t2 = time()
            print("Output shape: {}, Time taken: {}".format(out.shape, t2 - t1))
            feat_coll.append(out)
            label_coll.append(labels)

        out_features = torch.flatten(torch.stack(feat_coll), start_dim=0, end_dim=1)
        out_labels = torch.flatten(torch.stack(label_coll), start_dim=0, end_dim=1)

        print("The final features matrix has shape: {}".format(out_features.shape))

        if save_to_disk:
            # save as TensorDataset
            out_dataset = TensorDataset(out_features, out_labels)
            if train:
                prefix = "train"
            else:
                prefix = "test"
            filename = "{}_{}_dataset.pt".format(prefix, self.model.__class__.__name__)
            torch.save(out_dataset, filename)
            print("Saved features at {}/{}".format(os.getcwd(), filename))

        return out_features, out_labels

