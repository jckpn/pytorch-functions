# TODO: HAS NOT BEEN MODIFIED FROM PROJECT

import warnings

import random
import pandas as pd
import torch

warnings.simplefilter(action="ignore", category=FutureWarning)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path,
        output_type=torch.long,
        normalise_input_fn=lambda x: x,
        normalise_output_fn=lambda x: x,
        balance_rfs=False,
        noise=0.0,
    ):
        self.df = pd.read_csv(csv_path)
        self.output_type = output_type
        self.noise = noise

        # apply normalisation function
        # do this all at once here rather than every time getitem is called
        self.normalised_X = []
        self.normalised_y = []

        for entry_idx in range(len(self.df)):
            entry = self.df.iloc[entry_idx]
            old_X = entry[1:].values
            old_y = entry[0]
            new_X = normalise_input_fn(old_X)
            if new_X is None: continue
            new_y = normalise_output_fn(old_y)
            self.normalised_X.append(new_X)
            self.normalised_y.append(new_y)

        self.num_inputs = len(new_X)

        if balance_rfs:
            class_limit = 40
            num_classes = 10
            y_max = max(self.normalised_y)
            y_min = min(self.normalised_y)
            og_class_count = [0] * num_classes
            new_class_count = [0] * num_classes
            balanced_y = []
            balanced_X = []
            first_pass = True

            while min(new_class_count) < class_limit:
                for idx, y in enumerate(self.normalised_y):
                    # convert y (could be any value) to a class number 0-4
                    y_class = int((y - y_min) / (y_max - y_min) * num_classes)
                    if y_class > num_classes - 1: y_class = num_classes - 1

                    X = self.normalised_X[idx]
                    if first_pass: og_class_count[y_class] += 1
                    if new_class_count[y_class] < class_limit:
                        balanced_y.append(y)
                        balanced_X.append(X)
                        new_class_count[y_class] += 1
                first_pass = False

            self.normalised_y = balanced_y
            self.normalised_X = balanced_X

            print(og_class_count, 'balanced to', new_class_count)

    def __len__(self):
        return len(self.normalised_y)

    def __getitem__(self, idx):
        X = self.normalised_X[idx]
        y = self.normalised_y[idx]

        if isinstance(self.output_type, torch.dtype):
            if self.noise > 0.0:
                for i in range(len(X)):
                    X[i] = X[i] + random.uniform(-self.noise, self.noise)
                if self.output_type == torch.float32:
                    y = y + random.uniform(-self.noise, self.noise)  # add some noise to the output
            # convert to torch tensors
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=self.output_type)

        return X, y
