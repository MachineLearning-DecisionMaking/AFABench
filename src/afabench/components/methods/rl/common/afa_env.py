from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, final, override

import torch
import wandb
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from afabench.components.methods.rl.common.custom_types import (
    AFADatasetFn,
    AFARewardFn,
)
from afabench.core.types import AFAInitializeFn, AFAUnmaskFn

if TYPE_CHECKING:
    from afabench.core.types import (
        Features,
        Label,
    )


@final
class AFAEnv(EnvBase):
    """
    A dynamic-length MDP for active feature acquisition (AFA).

    The episode length is at most `hard_budget`, and the agent can choose to stop earlier.
    """

    @property
    @override
    def batch_locked(self) -> bool:
        return False

    @batch_locked.setter
    def batch_locked(self, value: bool) -> None:
        # AFAEnv doesn't support batch locking, so we ignore the setter
        pass

    def __init__(
        self,
        dataset_fn: AFADatasetFn,  # a function that returns data in batches when called
        reward_fn: AFARewardFn,
        device: torch.device | None,
        batch_size: torch.Size,
        feature_shape: torch.Size,
        n_selections: int,  # action dim = n_selections + 1 since we have a stop action as well
        n_classes: int,
        hard_budget: float
        | None,  # accumulated selection cost allowed before the episode ends. If None, no limit.
        initialize_fn: AFAInitializeFn,
        unmask_fn: AFAUnmaskFn,
        *,
        force_hard_budget: bool = False,  # if True and hard_budget is set, never allow the stop action
        seed: int | None = None,
        selection_costs: Sequence[float]
        | None = None,  # How much each sequence costs. If None, assume unit cost (1).
    ):
        # Do not allow empty batch sizes
        assert batch_size != torch.Size(()), "Batch size must be non-empty"
        assert len(batch_size) == 1, "Batch size must be 1D"
        super().__init__(device=device, batch_size=batch_size)

        self.dataset_fn = dataset_fn
        self.reward_fn = reward_fn
        self.feature_shape = feature_shape
        self.n_selections = n_selections
        self.n_classes = n_classes
        self.hard_budget = hard_budget
        if hard_budget is None:
            # If hard budget is not set, always allow agent to stop
            self.allow_stop_action = True
        else:
            # If hard budget is set, stop action is only allowed if force_hard_budget is false
            self.allow_stop_action = not force_hard_budget
        self.force_hard_budget = force_hard_budget
        self.initialize_fn = initialize_fn
        self.unmask_fn = unmask_fn
        self.seed = seed
        if selection_costs is None:
            self.selection_costs = torch.ones(
                (self.n_selections,), device=self.device
            )
        else:
            self.selection_costs = torch.tensor(
                selection_costs, device=self.device
            )

        self.rng = torch.Generator()
        if self.seed is not None:
            self.rng.manual_seed(self.seed)

        self._make_spec()

    def _make_spec(self) -> None:
        self.observation_spec = Composite(
            # For binary tensorspecs, torchrl now forces us to specify how large the last dimension is, I'm not sure why.
            feature_mask=Binary(
                n=self.feature_shape[-1],
                shape=self.batch_size + self.feature_shape,
                dtype=torch.bool,
            ),
            performed_action_mask=Binary(
                n=self.n_selections + 1,
                shape=self.batch_size + torch.Size((self.n_selections + 1,)),
                dtype=torch.bool,
            ),
            # "action" does include the stop action
            allowed_action_mask=Binary(
                n=self.n_selections + 1,
                shape=self.batch_size + torch.Size((self.n_selections + 1,)),
                dtype=torch.bool,
            ),
            # "selections" does not include the stop action
            performed_selection_mask=Binary(
                n=self.n_selections,
                shape=self.batch_size + torch.Size((self.n_selections,)),
                dtype=torch.bool,
            ),
            masked_features=Unbounded(
                shape=self.batch_size + self.feature_shape,
                dtype=torch.float32,
            ),
            # hidden from the agent
            features=Unbounded(
                shape=self.batch_size + self.feature_shape,
                dtype=torch.float32,
            ),
            label=Unbounded(
                shape=self.batch_size + (self.n_classes,),
                dtype=torch.float32,
            ),
            accumulated_cost=Unbounded(
                shape=self.batch_size, dtype=torch.float32
            ),
            batch_size=self.batch_size,
        )
        # One action per feature + stop action
        self.action_spec = Categorical(
            n=self.n_selections + 1,
            shape=self.batch_size + torch.Size(()),
            dtype=torch.int64,
        )
        self.reward_spec = Unbounded(
            shape=self.batch_size + torch.Size((1,)), dtype=torch.float32
        )
        self.done_spec = Binary(
            n=1, shape=self.batch_size + torch.Size((1,)), dtype=torch.bool
        )

    @override
    def _reset(
        self, tensordict: TensorDictBase | None, **_: dict[str, Any]
    ) -> TensorDict:
        if tensordict is None:
            tensordict = TensorDict(
                {}, batch_size=self.batch_size, device=self.device
            )

        # TorchRL calls _reset whenever *any* episode is done, passing a reset
        # mask of which ones, and then keeps only those entries of what we
        # return. Draw exactly that many training instances so that the
        # dataset pointer advances only by the instances actually consumed.
        reset_mask = tensordict.get("_reset", None)
        if reset_mask is None:
            return self._new_episodes(tensordict.batch_size, tensordict.device)

        reset_mask = reset_mask.reshape(tensordict.batch_size)
        new_episodes = self._new_episodes(
            torch.Size((int(reset_mask.sum()),)), tensordict.device
        )
        # Scatter the new episodes back to full batch shape. Entries outside
        # the reset mask are replaced by the ongoing episodes during TorchRL's
        # reset update, so their value does not matter.
        td = TensorDict(
            {
                key: value.new_zeros(tensordict.batch_size + value.shape[1:])
                for key, value in new_episodes.items()
            },
            batch_size=tensordict.batch_size,
            device=tensordict.device,
        )
        td[reset_mask] = new_episodes
        return td

    def _new_episodes(
        self, batch_size: torch.Size, device: torch.device | None
    ) -> TensorDict:
        """Start a batch of episodes on fresh training instances."""
        features, label = self.dataset_fn(batch_size)
        features: Features = features.to(device)
        label: Label = label.to(device)

        # Initialize features
        initial_feature_mask = self.initialize_fn(
            features=features, label=label, feature_shape=self.feature_shape
        )

        initial_masked_features = features.clone()
        initial_masked_features[~initial_feature_mask] = 0.0

        td = TensorDict(
            {
                "feature_mask": initial_feature_mask,
                "performed_action_mask": torch.zeros(
                    batch_size + torch.Size((self.n_selections + 1,)),
                    dtype=torch.bool,
                    device=device,
                ),
                "allowed_action_mask": torch.ones(
                    batch_size + torch.Size((self.n_selections + 1,)),
                    dtype=torch.bool,
                    device=device,
                ),
                "performed_selection_mask": torch.zeros(
                    batch_size + torch.Size((self.n_selections,)),
                    dtype=torch.bool,
                    device=device,
                ),
                "masked_features": initial_masked_features,
                "features": features,
                "label": label,
                "accumulated_cost": torch.zeros(
                    batch_size, dtype=torch.float32, device=device
                ),
            },
            batch_size=batch_size,
            device=device,
        )

        # If stop action is not allowed, disable it in the action mask
        if not self.allow_stop_action:
            td["allowed_action_mask"][:, 0] = False

        return td

    @override
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict["action"]
        stop_mask = action == 0
        selection = (action - 1).clamp(min=0)
        proposed_cost = torch.where(
            stop_mask, 0.0, self.selection_costs[selection]
        )

        # Forced stop, matching the evaluation loop: a selection whose cost
        # would push the accumulated cost past the hard budget is not
        # executed, and the episode ends with the state unchanged.
        forced_stop_mask = torch.zeros_like(stop_mask)
        if self.hard_budget is not None:
            forced_stop_mask = ~stop_mask & (
                tensordict["accumulated_cost"] + proposed_cost
                > self.hard_budget
            )

        # Unmask the whole batch and keep the old state where no selection is
        # executed, so there is no data-dependent indexing or host sync
        executes_selection = ~stop_mask & ~forced_stop_mask
        unmasked = self.unmask_fn(
            masked_features=tensordict["masked_features"],
            feature_mask=tensordict["feature_mask"],
            features=tensordict["features"],
            afa_selection=selection.unsqueeze(-1),
            selection_mask=tensordict["performed_selection_mask"],
            label=tensordict["label"],
            feature_shape=self.feature_shape,
        )
        new_feature_mask = torch.where(
            executes_selection.view(-1, *[1] * len(self.feature_shape)),
            unmasked,
            tensordict["feature_mask"],
        )
        new_masked_features = torch.where(
            new_feature_mask, tensordict["features"], 0.0
        )
        new_accumulated_cost = tensordict["accumulated_cost"] + torch.where(
            executes_selection, proposed_cost, 0.0
        )

        def mark(mask: Tensor, index: Tensor, value: Tensor) -> Tensor:
            index = index.unsqueeze(-1)
            return mask.scatter(-1, index, mask.gather(-1, index) | value)

        new_performed_action_mask = mark(
            tensordict["performed_action_mask"],
            action,
            (stop_mask | executes_selection).unsqueeze(-1),
        )
        new_performed_selection_mask = mark(
            tensordict["performed_selection_mask"],
            selection,
            executes_selection.unsqueeze(-1),
        )
        allowed = tensordict["allowed_action_mask"]
        index = action.unsqueeze(-1)
        new_allowed_action_mask = allowed.scatter(
            -1,
            index,
            allowed.gather(-1, index) & ~executes_selection.unsqueeze(-1),
        )

        # If stop action is not allowed, ensure it stays disabled
        if not self.allow_stop_action:
            new_allowed_action_mask[:, 0] = False

        # Done if we choose to stop (action 0), are forced to stop by the hard
        # budget, or all selection actions (1 through n_selections) are
        # exhausted
        selection_actions_available = new_allowed_action_mask[:, 1:].any(
            dim=-1
        )
        done = (
            stop_mask.unsqueeze(-1)
            | forced_stop_mask.unsqueeze(-1)
            | (~selection_actions_available).unsqueeze(-1)
        )

        # Always calculate a possible reward
        with torch.no_grad():
            reward = self.reward_fn(
                tensordict["masked_features"],
                tensordict["feature_mask"],
                tensordict["performed_selection_mask"],
                new_masked_features,
                new_feature_mask,
                new_performed_selection_mask,
                action,
                tensordict["features"],
                tensordict["label"],
                done,
            )

        r = TensorDict(
            {
                "performed_action_mask": new_performed_action_mask,
                "allowed_action_mask": new_allowed_action_mask,
                "performed_selection_mask": new_performed_selection_mask,
                "feature_mask": new_feature_mask,
                "masked_features": new_masked_features,
                "done": done,
                "reward": reward,
                # features and label are not cloned since they stay the same
                "features": tensordict["features"],
                "label": tensordict["label"],
                "accumulated_cost": new_accumulated_cost,
            },
            batch_size=tensordict.batch_size,
        )
        return r

    @override
    def _set_seed(self, seed: int | None) -> None:
        rng = torch.manual_seed(seed)
        self.rng = rng

    def get_batch_info(self, td: TensorDictBase) -> dict[str, Any]:
        """Return a wandb-loggable dictionary from a tensordict collected during training. Should only contain method-agnostic info."""
        return {
            "avg_reward": td["next", "reward"].mean().item(),
            # Average number of features selected when we stop
            "fraction observed at stop time": td["next", "feature_mask"][
                td["next", "done"].squeeze(-1)
            ]
            .float()
            .mean()
            .cpu()
            .item(),
        }

    def get_rollout_info(
        self, rollout_tds: list[TensorDictBase]
    ) -> dict[str, Any]:
        """Return a wandb-loggable dictionary from a lits of tensordicts collected during evaluation rollouts. Should only contain method-agnostic info."""
        # Every rollout td has shape (n_agents, episode_len)
        flat_td = torch.cat(rollout_tds, dim=-1).flatten()  # pyright: ignore[reportArgumentType, reportCallIssue]
        return {"action": wandb.Histogram(flat_td["action"].cpu())}
