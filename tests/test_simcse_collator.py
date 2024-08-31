from unittest.mock import Mock

import pandas as pd

from bg2vec.training import SimCSEDefaultCollator


def test_should_collate():
    # GIVEN
    model = Mock()
    model.tokenize.return_value = [1,2,3]
    collator = SimCSEDefaultCollator(model)

    features = [
        pd.DataFrame({"texts": ["a", "b", "c"], "label": [1, 2, 3]}),
        pd.DataFrame({"texts": ["d", "e", "f"], "label": [1, 2, 3]})

    ]
    # WHEN
    res = collator(features)

    # THEN
    assert res is not None