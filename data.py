# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import random

import numpy as np
import pandas as pd
import paddle
from paddlenlp.datasets import MapDataset

class_code = {
    "火灾扑救": 1,
    "抢险救援": 2,
    "社会救助": 3
}


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens. And creates a mask from the two sequences passed 
    to be used in a sequence-pair classification task.
        
    A BERT sequence has the following format:

    - single sequence: ``[CLS] X [SEP]``
    - pair of sequences: ``[CLS] A [SEP] B [SEP]``

    A BERT sequence pair mask has the following format:
    ::
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |

    If only one sequence, only returns the first portion of the mask (0's).


    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): List of sequence pair mask.
        label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
    """
    encoded_inputs = tokenizer(text=example["text"], max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array(example["label"], dtype="int64")
        return input_ids, token_type_ids, label
    return input_ids, token_type_ids


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)


def read_custom_data(filename, is_test=False):
    """Reads data."""
    data = pd.read_csv(filename)
    for line in data.values:
        if is_test:
            text = line[1]
            yield {"text": clean_text(text), "label": ""}
        else:
            text, label = line[1], line[2:]
            yield {"text": clean_text(text), "label": label}


def read_excel_data(filename, is_test=False):
    """Reads data."""
    data = pd.read_excel(filename)
    for index, line in data.iterrows():
        if is_test:
            text = line['JQNR']
            yield {"text": clean_text(text), "label": ""}
        else:
            text, label = line['JQNR'], line['JQLX']
            label_code = class_code[label] if label in class_code else 0
            yield {"text": clean_text(text), "label": label_code}


def clean_text(text):
    text = text.replace("\r", "").replace("\n", "")
    text = re.sub(r"\\n\n", ".", text)
    return text


def write_test_results(filename, results, label_info):
    """write test results"""
    data = pd.read_csv(filename)
    qids = [line[0] for line in data.values]
    results_dict = {k: [] for k in label_info}
    results_dict["id"] = qids
    results = list(map(list, zip(*results)))
    for key in results_dict:
        if key != "id":
            for result in results:
                results_dict[key] = result
    df = pd.DataFrame(results_dict)
    df.to_csv("sample_test.csv", index=False)
    print("Test results saved")


class DataProcessor(object):
    """Base class for data converters for sequence classification datasets."""

    def __init__(self, negative_num=1):
        # Random negative sample number for efl strategy
        self.neg_num = negative_num

    def get_train_datasets(self, datasets, task_label_description):
        """See base class."""
        return self._create_examples(datasets, "train", task_label_description)

    def get_dev_datasets(self, datasets, task_label_description):
        """See base class."""
        return self._create_examples(datasets, "dev", task_label_description)

    def get_test_datasets(self, datasets, task_label_description):
        """See base class."""
        return self._create_examples(datasets, "test", task_label_description)


class FireProcessor(DataProcessor):
    """Processor for the fire dataset."""

    def _create_examples(self, datasets, phase, task_label_description):
        """Creates examples for the training and dev sets."""

        examples = []
        if phase == "train":
            for example in datasets:
                true_label = str(example["label"])
                neg_examples = []
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence']
                    new_example["sentence2"] = label_description

                    # Todo: handle imbalanced example, maybe hurt model performance
                    if true_label == label:
                        new_example["label"] = 1
                        examples.append(new_example)
                    else:
                        new_example["label"] = 0
                        neg_examples.append(new_example)
                neg_examples = random.sample(neg_examples, self.neg_num)
                examples.extend(neg_examples)

        elif phase == "dev":
            for example in datasets:
                true_label = str(example["label"])
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence']
                    new_example["sentence2"] = label_description

                    # Get true_label's index at task_label_description for evaluate
                    true_label_index = list(task_label_description.keys()).index(true_label)
                    new_example["label"] = true_label_index
                    examples.append(new_example)

        elif phase == "test":
            for example in datasets:
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence']
                    new_example["sentence2"] = label_description
                    examples.append(new_example)

        return MapDataset(examples)


processor_dict = {
    "fire": FireProcessor
}
