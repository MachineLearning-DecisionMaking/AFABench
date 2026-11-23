import pytest
import torch
from torch import nn

from afabench.components.methods.rl.odin.models import PointNet, PointNetType


@pytest.mark.parametrize(
    "pointnet_type", [PointNetType.POINTNET, PointNetType.POINTNETPLUS]
)
def test_adding_an_element_adds_its_feature_map(
    pointnet_type: PointNetType,
) -> None:
    torch.manual_seed(0)
    n_features, identity_size = 8, 4
    in_size = identity_size + (pointnet_type == PointNetType.POINTNET)
    pointnet = PointNet(
        identity_size,
        n_features,
        nn.Sequential(nn.Linear(in_size, 16), nn.ReLU(), nn.Linear(16, 3)),
        pointnet_type,
    )
    x = torch.randn(5, n_features)
    mask = torch.rand(5, n_features) < 0.5
    mask[:, 2] = False

    base = pointnet(x * mask, mask)
    added = mask.clone()
    added[:, 2] = True
    expected = pointnet(x * added, added)
    actual = base + pointnet.element_maps(x)[:, 2]
    assert torch.allclose(actual, expected, atol=1e-6)
