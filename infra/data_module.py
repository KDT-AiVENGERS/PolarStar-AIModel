# Modules About Hydra
from typing import List, Any

# Modules About Torch
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Modules About Pytorch Lightning
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, STEP_OUTPUT

# Modules About Pandas, Matplotlib, Numpy
import pandas as pd
import numpy as np

# Others
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


'''
Input
data: {'input_ids': [tensor1, tensor2, ...], 'token_type_ids': [tensor1, tensor2, ...], 'attention_mask': [tensor1, tensor2, ...]}
'''
class CustomDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()

        self.data = data
        self.keys = list(data.keys())

    def __len__(self):
        return len(self.data[self.keys[0]])

    def __getitem__(self, index) -> Any:
        item = []
        for key in self.keys:
            item.append(self.data[key][index])

        return item

class HFBertDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        max_batch_size=64,
        data_path='./data/JobDescription.pre_result_2.csv',
        predict_target_cols=[],
        train_target_cols=[],
        max_length=None,
        sliding_window_interval=200,
        train_test_ratio=0.9,
        train_val_ratio=0.8,
        masked_token_ratio=0.15
    ) -> None:
        super().__init__()
        self.predict_target_cols = predict_target_cols
        self.train_target_cols = train_target_cols
        self.data_path = data_path

        self.train_test_ratio = train_test_ratio
        self.train_val_ratio = train_val_ratio

        self.batch_size = max_batch_size
        if predict_target_cols:
            self.predict_batch_size = int(max_batch_size / len(predict_target_cols)) * len(predict_target_cols)

        # load Bert Tokenizer
        self.tokenizer = tokenizer

        if max_length:
            self.max_length = max_length
        else:
            self.max_length = tokenizer.model_max_length

        self.sliding_window_interval = sliding_window_interval

        self.masked_token_ratio = masked_token_ratio

    def prepare_data(self) -> None:
        # load predict data
        try:
            self.predict_data_pd = pd.read_csv(self.data_path)
        except:
            print('No inference data available!')

        if self.predict_data_pd is not None and self.predict_target_cols:
            self.predict_data_pd = self.predict_data_pd.sample(30) # for server test
            # serialize columns
            predict_data_serialized = []
            for row in range(len(self.predict_data_pd)):
                for col in self.predict_target_cols:
                    predict_data_serialized.append(self.predict_data_pd.iloc[row][col])

            # make tokens
            self.predict_tokens = self.tokenizer(predict_data_serialized, return_tensors='pt', padding=True, truncation=True)

            # make predict dataset
            self.predict_dataset = CustomDataset(self.predict_tokens)
            self.predict_token_keys = self.predict_tokens.keys()

    def setup(self, stage: str) -> None:
        # load train data
        try:
            self.train_data_pd = pd.read_csv(self.data_path)
        except:
            print('No training data available!')
            self.train_data_pd = None

        if self.train_data_pd is not None and self.train_target_cols:
            # serialize columns
            train_data_serialized = []
            for col in self.train_target_cols:
                train_data_serialized += list(self.train_data_pd[col])

            # make tokens
            self.train_tokens = self.tokenizer(train_data_serialized, return_tensors='pt', padding=True)
            self.train_token_keys = self.train_tokens.keys()

            # slicing tokens by a sliding window
            current_token_length = self.train_tokens['input_ids'].shape[1]
            if current_token_length > self.max_length:
                self.train_tokens_sliced = self._make_sliced_tokens(self.train_tokens, current_token_length)
            else:
                self.train_tokens_sliced = self.train_tokens

            # make train dataset
            train_dataset = CustomDataset(self.train_tokens_sliced)

            # split train val test datasets
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                train_dataset,
                [
                    self.train_test_ratio * self.train_val_ratio,
                    self.train_test_ratio * (1 - self.train_val_ratio),
                    1 - self.train_test_ratio
                ]
            )

    def _collate_fn_predict(self, batch):
        '''
        Inputs
        batch: [[tensor1_1, tensor1_2, tensor1_3], [tensor2_1, tensor2_2, tensor2_3], ...]
        self.predict_token_keys: ['input_ids', 'token_type_ids', 'attention_mask']

        Output
        dict_by_keys: {'input_ids': [tensor1_1, tensor2_1, ...], 'token_type_ids': [tensor1_2, tensor2_2, ...], 'attention_mask': [tensor1_3, tensor2_3, ...]}
        '''
        list_by_keys = list(zip(*batch))
        dict_by_keys = {}
        for i, key in enumerate(self.predict_token_keys):
            dict_by_keys[key] = torch.stack(list_by_keys[i])

        return dict_by_keys

    def _collate_fn_train(self, batch):
        list_by_keys = list(zip(*batch))
        dict_by_keys = {}
        for i, key in enumerate(self.train_token_keys):
            dict_by_keys[key] = torch.stack(list_by_keys[i])

        dict_by_keys['labels'] = dict_by_keys['input_ids'].clone()

        for i, tokens in enumerate(dict_by_keys['input_ids']):
            self._make_tokens(tokens, dict_by_keys['labels'][i], self.masked_token_ratio)

        return dict_by_keys

    def _make_sliced_tokens(self, tokens, tokens_length):
        train_tokens_sliced = {}
        for key in self.train_token_keys:
            train_tokens_sliced[key] = []

        for i in range(len(tokens[key])):
            window_index = 0
            while True:
                if window_index + self.max_length <= tokens_length:
                    for key in self.train_token_keys:
                        train_tokens_sliced[key].append(tokens[key][i][window_index:window_index + self.max_length])

                    if tokens[key][i][window_index + self.max_length - 1] != self.tokenizer.pad_token_id:
                        window_index += self.sliding_window_interval
                        continue
                break

        return train_tokens_sliced

    def _make_tokens(self, tensor1, tensor2, mask_token_ratio, masking_ratio=[0.8, 0.1, 0.1]):
        assert sum(masking_ratio) == 1

        token_len = 0
        for token in tensor1:
            if token != self.tokenizer.pad_token_id:
                token_len += 1
                continue
            break

        masked_tokens = torch.tensor(np.random.choice(range(1, token_len - 1), int((token_len - 2) * mask_token_ratio)), dtype=torch.int)
        token_types = torch.randint_like(masked_tokens, 1, 101)

        tensor2[:] = -100
        tensor2[masked_tokens] = tensor1[masked_tokens]
        tensor1[masked_tokens] = torch.where(token_types <= int(masking_ratio[0] * 100), self.tokenizer.mask_token_id, torch.where(token_types <= int((1 - masking_ratio[2]) * 100), torch.randint(0, self.tokenizer.vocab_size, (len(masked_tokens),)), tensor1[masked_tokens]))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self._collate_fn_train)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self._collate_fn_train)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self._collate_fn_train)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.predict_dataset, batch_size=self.predict_batch_size, collate_fn=self._collate_fn_predict)