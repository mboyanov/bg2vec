from unittest.mock import Mock

import pandas as pd
from llm2vec.dataset.dataset import TrainSample

from bg2vec.training import SimCSEDefaultCollator


def test_should_collate():
    # GIVEN
    model = Mock()
    model.tokenize.return_value = [1,2,3]
    collator = SimCSEDefaultCollator(model)

    features = [
        TrainSample(texts=["I am an apple", "I am an orange"], label=1),
        TrainSample(texts=["I like to move it move it", "I like to shake it shake it"], label=1),

    ]
    # WHEN
    res = collator(features)

    # THEN
    assert res is not None