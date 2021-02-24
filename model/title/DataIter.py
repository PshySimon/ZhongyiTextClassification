"""
@Author:    Pshy Simon
@Date:  2020/10/20 0020 下午 02:33
@Description:
    数据迭代器
"""
import json
import logging
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from math import floor
import gc


class DataIter:

    def __init__(self, config):
        self.config = config
        self.train_path = "./train_v1.csv"
        self.test_path = "./test_content_v1.csv"
        self.train_df = None
        self.dev_df = None
        self.test_df = None
        self._data = []
        self.preprocessor()

    def preprocessor(self):
        self.train_df = pd.read_csv(self.train_path, sep="\t")
        self.test_df = pd.read_csv(self.test_path, sep="\t")

    def build_examples(self, raw_data, test = False):           # 需要构建四个特征
        tokens, input_ids, attention_masks = [], [], []

        def convert_input_to_ids(inputs):
            tokens = self.config.tokenizer.tokenize(inputs)[:self.config.max_length]      # tokenize

            text_len = len(tokens)
            ids = self.config.tokenizer.convert_tokens_to_ids(
                tokens + ["[PAD]"]*(self.config.max_length - text_len))
            att_mask = [1] * text_len + [0] * (self.config.max_length - text_len)

            return ids, att_mask

        labels = None
        if not test:
            labels = raw_data.label.tolist()

        for i, row in tqdm(raw_data.iterrows()):
            input_id, attention_mask = convert_input_to_ids(str(row['content']))
            input_ids.append(input_id)
            attention_masks.append(attention_mask)

        return tuple(np.asarray(x, dtype=np.int32) for x in(input_ids, attention_masks, labels) if x is not None)

        

    def _to(self, tensor):
        # 记录下传到gpu的张量，便于回收
        res = torch.tensor(tensor, dtype=torch.long).to(self.config.device)
        self._data.append(res)
        return res

    def _gc(self):
        # 清空并回收数据
        for x in self._data:
            x.cpu()
            torch.cuda.empty_cache()
            del x
        gc.collect()

    # 根据多折验证传过来的训练集和验证集，包装到数据集中
    def build_dataset(self, train_data, dev_data):
        logging.warning(msg="Loading data from storage, it may cost a time.")
        train = TensorDataset(*tuple(self._to(x) for x in train_data))
        dev = TensorDataset(*tuple(self._to(x) for x in dev_data))
        return train, dev

    def build_test(self):
        test_data = self.build_examples(self.test_df, test = True)
        test = TensorDataset(*tuple(self._to(x) for x in test_data))
        return DataLoader(test, batch_size=self.config.batch_size)

    def build_iter(self, data):
        df = TensorDataset(*tuple(self._to(x) for x in data))
        return DataLoader(df, batch_size=self.config.batch_size)    

    def build_iterator(self, train_data, dev_data):
        logging.warning(msg="Building dataset...")
        train, dev = self.build_dataset(train_data, dev_data)
        return (DataLoader(x, batch_size=self.config.batch_size, shuffle=True) if x is not None else None for x in (train, dev))




