import functools
from unittest import mock

import datasets
import pandas as pd
from transformers import TrainingArguments, PreTrainedTokenizer

from bg2vec.arguments import DataTrainingArguments
from bg2vec.preprocessing import tokenize_datasets, group_texts


def test_should_tokenize():
    # GIVEN
    raw_datasets = datasets.DatasetDict({
        "train": datasets.Dataset.from_pandas(pd.DataFrame({"text": ["hello world"]}))
    })
    # WHEN
    tokenizer = mock.MagicMock(spec=PreTrainedTokenizer)
    tokenizer.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
    data_args = DataTrainingArguments(dataset_name="dummy")
    training_args = TrainingArguments(output_dir="/tmp")
    tokenized_datasets = tokenize_datasets(raw_datasets, tokenizer, data_args, training_args, max_seq_length=1024)

    # THEN
    assert tokenized_datasets is not None
    assert tokenized_datasets['train'][0] == {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    assert 'text' not in tokenized_datasets['train'].column_names


def test_should_group():
    # GIVEN
    tokenized_dataset = datasets.Dataset.from_pandas(pd.DataFrame({"input_ids": [[1, 2, 3], [1, 2, 3, 4,5]]}))
    valid_dataset = datasets.Dataset.from_pandas(pd.DataFrame({"input_ids": [[1, 2, 3], [4,5,6]]}))
    tokenized_datasets = datasets.DatasetDict({"train": tokenized_dataset, "valid":valid_dataset})

    group_fn = functools.partial(group_texts, max_seq_length=6)

    # WHEN
    grouped_datasets = tokenized_datasets.map(group_fn, batched=True)

    # THEN
    assert grouped_datasets is not None
    assert len(grouped_datasets['train']) == 1
    assert grouped_datasets['train']['input_ids'][0] == [1, 2, 3, 1, 2, 3]
    assert grouped_datasets['valid']['input_ids'][0] == [1, 2, 3, 4, 5, 6]