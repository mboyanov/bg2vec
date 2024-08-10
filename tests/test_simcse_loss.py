from llm2vec.loss import HardNegativeNLLLoss
import torch

def test_should_handle_no_negative_samples():
    # GIVEN
    loss = HardNegativeNLLLoss()

    # WHEN

    q_reps = torch.rand(10, 768)
    d_reps_pos = torch.rand(10, 768)
    l = loss(q_reps, d_reps_pos)

    # THEN
    assert l is not None