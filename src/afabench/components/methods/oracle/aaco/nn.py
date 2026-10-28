import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, final, override

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm

from afabench.components.initializers.random_initializer import (
    RandomInitializer,
)
from afabench.components.methods.common import AFAMethodBase, on_method_device
from afabench.components.methods.oracle.aaco.afa_methods import AACOAFAMethod
from afabench.components.methods.oracle.aaco.utils import (
    compute_patch_selection_mask,
    ensure_probabilities,
    flatten_for_aaco,
    uses_patch_selection,
)
from afabench.core.bundle_system.bundle import load_bundle
from afabench.core.types import (
    AFAAction,
    AFAClassifier,
    AFAInitializer,
    AFAUnmasker,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)

logger = logging.getLogger(__name__)


def _build_selection_mask(
    feature_mask_structured: torch.Tensor,
    selection_size: int,
    feature_shape: torch.Size,
    device: torch.device,
) -> torch.Tensor:
    feature_mask_flat = feature_mask_structured.reshape(-1)
    if selection_size == feature_mask_flat.numel():
        return feature_mask_flat.bool().clone()
    if uses_patch_selection(selection_size, feature_shape):
        return compute_patch_selection_mask(
            feature_mask_structured, selection_size, feature_shape
        ).squeeze(0)
    return torch.zeros(selection_size, dtype=torch.bool, device=device)


def generate_aaco_rollouts(
    aaco_method: AACOAFAMethod,
    features: torch.Tensor,
    labels: torch.Tensor,
    feature_shape: torch.Size | None = None,
    unmasker: AFAUnmasker | None = None,
    initializer: AFAInitializer | None = None,
    max_acquisitions: int | None = None,
    device: torch.device | None = None,
    rollout_batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate expert rollouts from an AACO oracle.

    For each sample, we simulate the AACO acquisition process and collect
    (state, action) pairs at each step.

    Args:
        aaco_method: Trained AACOAFAMethod instance
        features: Training features (N x d)
        labels: Training labels (N x n_classes), one-hot encoded
        feature_shape: Shape of features excluding batch dim
        unmasker: Unmasker to apply selections
        initializer: Initializer used for rollout start states
        max_acquisitions: Maximum acquisitions per sample (None = until oracle stops)
        device: Device to use

    Returns:
        Tuple of (masked_features, feature_masks, actions):
        - masked_features: (total_steps, d) - observed features at each step
        - feature_masks: (total_steps, d) - boolean mask of observed features
        - actions: (total_steps,) - action taken (0=stop, 1-d=feature to acquire)
    """
    del labels  # Unused, kept for API compatibility
    if device is None:
        device = features.device
    assert isinstance(aaco_method, AACOAFAMethod)

    n_samples, n_features = features.shape
    if feature_shape is None:
        feature_shape = torch.Size([n_features])

    assert unmasker is not None, (
        "AACO+NN rollouts require an unmasker to apply selections."
    )

    selection_size = unmasker.get_n_selections(feature_shape=feature_shape)

    if initializer is None:
        initializer = RandomInitializer(num_initial_features=1)

    logger.info(f"Generating rollouts for {n_samples} samples...")

    # Step a chunk of samples together. Each sample's (state, action) pairs
    # are recorded with its index and step, then restored to sample-major
    # order so the output matches generating one sample at a time.
    records: list[tuple[torch.Tensor, ...]] = []
    for start in tqdm(
        range(0, n_samples, rollout_batch_size),
        desc="Generating AACO rollouts",
        unit="batch",
    ):
        idx = torch.arange(
            start, min(start + rollout_batch_size, n_samples), device=device
        )
        x_view = features[idx.cpu()].to(device).view(len(idx), *feature_shape)
        feature_mask = initializer.initialize(
            features=x_view, label=None, feature_shape=feature_shape
        ).to(device)
        selection_mask = torch.stack(
            [
                _build_selection_mask(m, selection_size, feature_shape, device)
                for m in feature_mask
            ]
        )
        active = torch.arange(len(idx), device=device)
        step = 0
        while len(active) > 0:
            mask = feature_mask[active]
            masked = x_view[active] * mask.float()
            if max_acquisitions is not None and step >= max_acquisitions:
                action = torch.zeros(
                    len(active), dtype=torch.long, device=device
                )
            else:
                action = aaco_method.act_for_instances(
                    masked_features=masked.flatten(1),
                    feature_mask=mask.flatten(1).float(),
                    selection_mask=selection_mask[active],
                    feature_shape=feature_shape,
                    split_indices=idx[active],
                ).view(-1)
            records.append(
                (
                    idx[active] * (n_features + 2) + step,
                    masked.flatten(1),
                    mask.flatten(1),
                    action,
                )
            )
            go = action != 0
            rows = active[go]
            if len(rows) > 0:
                selection = action[go] - 1
                selection_mask[rows, selection] = True
                feature_mask[rows] = unmasker.unmask(
                    masked_features=x_view[rows] * feature_mask[rows].float(),
                    feature_mask=feature_mask[rows],
                    features=x_view[rows],
                    afa_selection=selection.unsqueeze(-1),
                    selection_mask=selection_mask[rows],
                    label=None,
                    feature_shape=feature_shape,
                ).to(device)
            active = rows
            step += 1

    order = torch.cat([r[0] for r in records]).argsort()
    all_masked_features = torch.cat([r[1] for r in records])[order]
    all_feature_masks = torch.cat([r[2] for r in records])[order]
    all_actions = torch.cat([r[3] for r in records])[order]

    logger.info(
        "Generated %s state-action pairs from %s samples",
        len(all_actions),
        n_samples,
    )
    logger.info(
        f"  Average steps per sample: {len(all_actions) / n_samples:.2f}"
    )

    return all_masked_features, all_feature_masks, all_actions


class AACOPolicyNetwork(nn.Module):
    """
    Neural network policy for AACO+NN.

    Takes (masked_features, feature_mask) as input and outputs action logits.
    Architecture follows the masked MLP classifier pattern.
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,  # n_features + 1 (including stop action)
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        # Input: concatenated (masked_features, feature_mask)
        input_dim = n_features * 2

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_actions))

        self.network: nn.Sequential = nn.Sequential(*layers)
        self.n_features: int = n_features
        self.n_actions: int = n_actions

    @override
    def forward(
        self,
        masked_features: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            masked_features: (batch, d) observed features (unobserved = 0)
            feature_mask: (batch, d) boolean mask of observed features

        Returns:
            action_logits: (batch, n_actions) logits for each action
        """
        # Concatenate masked features and mask
        x = torch.cat([masked_features, feature_mask.float()], dim=-1)
        return self.network(x)


def _run_train_epoch(
    policy_network: AACOPolicyNetwork,
    train_loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float]:
    """Run a single training epoch and return (loss, accuracy)."""
    policy_network.train()
    total_loss, correct, total = 0.0, 0, 0
    for features_cpu, masks_cpu, actions_cpu in train_loader:
        features = features_cpu.to(device)
        masks = masks_cpu.to(device)
        actions = actions_cpu.to(device)
        optimizer.zero_grad()
        logits = policy_network(features, masks)
        loss = criterion(logits, actions)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(actions)
        correct += (logits.argmax(dim=-1) == actions).sum().item()
        total += len(actions)
    return total_loss / total, correct / total


def _run_val_epoch(
    policy_network: AACOPolicyNetwork,
    val_loader: DataLoader[Any],
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float]:
    """Run a single validation epoch and return (loss, accuracy)."""
    policy_network.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for features_cpu, masks_cpu, actions_cpu in val_loader:
            features = features_cpu.to(device)
            masks = masks_cpu.to(device)
            actions = actions_cpu.to(device)
            logits = policy_network(features, masks)
            loss = criterion(logits, actions)
            total_loss += loss.item() * len(actions)
            correct += (logits.argmax(dim=-1) == actions).sum().item()
            total += len(actions)
    return total_loss / total, correct / total


def train_policy_network(
    policy_network: AACOPolicyNetwork,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    max_epochs: int,
    learning_rate: float,
    patience: int,
    device: torch.device,
    metric_logger: Callable[[dict[str, float]], None] | None = None,
) -> AACOPolicyNetwork:
    """Train the policy network via behavioral cloning."""
    policy_network = policy_network.to(device)
    optimizer = torch.optim.AdamW(
        policy_network.parameters(), lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state_dict = None

    for epoch in range(max_epochs):
        train_loss, train_acc = _run_train_epoch(
            policy_network, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = _run_val_epoch(
            policy_network, val_loader, criterion, device
        )

        logger.info(
            f"Epoch {epoch + 1}/{max_epochs}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )
        if metric_logger is not None:
            metric_logger(
                {
                    "aaco_nn/epoch": float(epoch + 1),
                    "aaco_nn/train_loss": train_loss,
                    "aaco_nn/train_accuracy": train_acc,
                    "aaco_nn/val_loss": val_loss,
                    "aaco_nn/val_accuracy": val_acc,
                }
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state_dict = policy_network.state_dict().copy()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state_dict is not None:
        policy_network.load_state_dict(best_state_dict)
        logger.info(
            f"Loaded best model with validation loss {best_val_loss:.4f}"
        )

    return policy_network


def create_rollout_data_loaders(
    masked_features: torch.Tensor,
    feature_masks: torch.Tensor,
    actions: torch.Tensor,
    batch_size: int,
    val_split: float,
    seed: int,
) -> tuple[DataLoader[Any], DataLoader[Any], int, int]:
    """Create train and validation data loaders from rollout data."""
    tensor_dataset = TensorDataset(
        masked_features.cpu(), feature_masks.cpu().float(), actions.cpu()
    )
    val_size = int(len(tensor_dataset) * val_split)
    train_size = len(tensor_dataset) - val_size

    train_dataset, val_dataset = random_split(
        tensor_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, len(train_dataset), len(val_dataset)


@dataclass
@final
class AACONNAFAMethod(AFAMethodBase):
    """
    AACO+NN: Neural network approximation of AACO.

    This method uses a trained neural network policy to select features,
    providing much faster inference than the KNN-based AACO oracle while
    maintaining similar acquisition quality.
    """

    policy_network: AACOPolicyNetwork
    classifier: AFAClassifier
    dataset_name: str
    classifier_bundle_path: Path | None = None
    _device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    force_acquisition: bool = False

    def __post_init__(self):
        self.to(self._device)

    @override
    @on_method_device
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        """
        Select next feature to acquire using the neural network policy.

        Args:
            masked_features: Currently observed features (unobserved = 0)
            feature_mask: Boolean mask of observed features
            selection_mask: Which selections have been made (unused)
            label: True label (unused)
            feature_shape: Shape of features excluding batch dim

        Returns:
            AFAAction tensor with shape (*batch, 1):
            - 0 = stop acquiring
            - 1 to N = 1-indexed feature to acquire
        """
        del label  # Unused
        masked_features, feature_mask, batch_shape = flatten_for_aaco(
            masked_features, feature_mask, feature_shape
        )
        n_features = masked_features.shape[-1]

        # Get policy network predictions
        self.policy_network.eval()
        with torch.no_grad():
            action_logits = self.policy_network(masked_features, feature_mask)
            # Shape: (batch, n_actions) where n_actions = n_features + 1

        # Apply mask to already-acquired features (can't re-acquire)
        # Action 0 = stop, actions 1-n = acquire feature 0 to n-1
        # Set logits to -inf for already acquired features
        selection_size = self.policy_network.n_actions - 1
        selection_mask_flat = (
            selection_mask.view(-1, selection_size).bool()
            if selection_mask is not None
            else None
        )

        if selection_size == n_features:
            acquired_mask = feature_mask.bool()
        elif uses_patch_selection(selection_size, feature_shape):
            assert feature_shape is not None
            fm = feature_mask.view(-1, *feature_shape)
            acquired_mask = compute_patch_selection_mask(
                fm, selection_size, feature_shape
            )
        else:
            acquired_mask = torch.zeros(
                (masked_features.shape[0], selection_size),
                dtype=torch.bool,
                device=self._device,
            )

        if selection_mask_flat is not None:
            acquired_mask = acquired_mask | selection_mask_flat

        action_logits[:, 1:][acquired_mask] = float("-inf")

        if self.force_acquisition:
            has_available = ~acquired_mask.all(dim=-1)
            action_logits[has_available, 0] = float("-inf")

        # Select best action
        actions = action_logits.argmax(dim=-1)

        # Reshape to match batch shape
        return actions.view(*batch_shape, 1)

    @override
    @on_method_device
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        """
        Make prediction using the classifier.

        Args:
            masked_features: Currently observed features
            feature_mask: Boolean mask of observed features
            label: True label (unused)
            feature_shape: Shape of features excluding batch dim

        Returns:
            Class probabilities with shape (*batch, n_classes)
        """
        del label  # Unused
        masked_features_flat, feature_mask_flat, batch_shape = (
            flatten_for_aaco(masked_features, feature_mask, feature_shape)
        )

        # Get classifier predictions
        classifier_shape = torch.Size([masked_features_flat.shape[1]])
        with torch.no_grad():
            logits = self.classifier(
                masked_features_flat,
                feature_mask_flat,
                feature_shape=classifier_shape,
            )
            probs = ensure_probabilities(logits)

        # Reshape and return
        n_classes = probs.shape[-1]
        output_shape = batch_shape + (n_classes,)
        return probs.view(*output_shape)

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        return False

    @property
    @override
    def cost_param(self) -> float:
        """Return a dummy cost param (not used by AACO+NN)."""
        return 0.0

    @override
    def set_cost_param(self, cost_param: float) -> None:
        """Set cost param (no-op for AACO+NN, budget is implicit in training)."""

    @override
    def save(self, path: Path) -> None:
        """Save method state to disk."""
        state = {
            "policy_network_state_dict": self.policy_network.state_dict(),
            "n_features": self.policy_network.n_features,
            "n_actions": self.policy_network.n_actions,
            "dataset_name": self.dataset_name,
            "force_acquisition": self.force_acquisition,
            "classifier_bundle_path": str(self.classifier_bundle_path)
            if self.classifier_bundle_path is not None
            else None,
        }
        torch.save(state, path / f"aaco_nn_{self.dataset_name}.pt")
        logger.info(f"Saved AACO+NN method to {path}")

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device | None = None) -> Self:
        """Load method from disk."""
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        nn_files = list(path.glob("aaco_nn_*.pt"))
        if not nn_files:
            msg = f"No AACO+NN files found in {path}"
            raise FileNotFoundError(msg)

        state = torch.load(
            nn_files[0], map_location=device, weights_only=False
        )

        # Reconstruct policy network
        policy_network = AACOPolicyNetwork(
            n_features=state["n_features"],
            n_actions=state["n_actions"],
        )
        policy_network.load_state_dict(state["policy_network_state_dict"])

        # Load classifier from saved bundle path
        classifier_bundle_path = state.get("classifier_bundle_path")
        assert classifier_bundle_path is not None, (
            "AACO+NN loading requires classifier_bundle_path in saved state."
        )

        classifier_bundle_path = Path(classifier_bundle_path)
        classifier = load_bundle(
            classifier_bundle_path, expect=AFAClassifier, device=device
        ).obj

        force_acquisition = state.get("force_acquisition")
        if force_acquisition is None:
            force_acquisition = state.get("hard_budget") is not None

        method = cls(
            policy_network=policy_network,
            classifier=classifier,
            dataset_name=state["dataset_name"],
            classifier_bundle_path=classifier_bundle_path,
            force_acquisition=force_acquisition,
            _device=device,
        )

        return method


def create_aaco_nn_method(
    policy_network: AACOPolicyNetwork,
    classifier: AFAClassifier,
    dataset_name: str,
    classifier_bundle_path: Path | None = None,
    *,
    force_acquisition: bool = False,
    device: torch.device | None = None,
) -> AACONNAFAMethod:
    """
    Create an AACO+NN method instance.

    Args:
        policy_network: Trained policy network
        classifier: Classifier for predictions
        dataset_name: Name of dataset
        classifier_bundle_path: Path to classifier bundle (for save/load)
        force_acquisition: If True, never stop early (hard budget mode)
        device: Device to use

    Returns:
        Configured AACONNAFAMethod instance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return AACONNAFAMethod(
        policy_network=policy_network,
        classifier=classifier,
        dataset_name=dataset_name,
        classifier_bundle_path=classifier_bundle_path,
        force_acquisition=force_acquisition,
        _device=device,
    )
