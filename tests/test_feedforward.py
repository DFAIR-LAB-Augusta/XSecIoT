import torch
import torch.nn as nn

from firce.models.feedforward_binary import FeedForwardBinary


def test_feedforward_binary_forward_shape():
    model = FeedForwardBinary(input_dim=5)
    x = torch.randn(8, 5)
    out = model(x)
    assert out.shape == (8,)


def test_feedforward_binary_training_step_updates_weights():
    torch.manual_seed(0)
    model = FeedForwardBinary(input_dim=5)
    before = [p.clone() for p in model.parameters()]

    x = torch.randn(16, 5)
    y = torch.randint(0, 2, (16,), dtype=torch.float32)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.BCEWithLogitsLoss()

    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
