import pytest
import torch

from afabench.components.methods.rl.ol.config import OLPQModuleConfig
from afabench.components.methods.rl.ol.models import OLPQModule


@pytest.mark.parametrize("use_feature_mask", [False, True])
def test_confidence_matches_full_forward_on_repeated_inputs(
    *, use_feature_mask: bool
) -> None:
    torch.manual_seed(0)
    module = OLPQModule(
        n_features=10,
        n_classes=3,
        n_actions=11,
        cfg=OLPQModuleConfig(
            n_hiddens=[16, 8, 4],
            p_dropout=0.5,
            use_feature_mask=use_feature_mask,
        ),
    )
    mask = torch.rand(4, 10) < 0.5
    x = torch.randn(4, 10) * mask
    samples = 7

    torch.manual_seed(1)
    logits, _ = module(
        module._repeat_batch(x, samples),  # noqa: SLF001
        module._repeat_batch(mask.float(), samples),  # noqa: SLF001
        mc_dropout=True,
    )
    expected = logits.view(4, samples, -1).softmax(dim=-1).mean(dim=1)

    torch.manual_seed(1)
    actual = module.confidence(x, samples, mask.float())
    assert torch.allclose(actual, expected, atol=1e-6)
