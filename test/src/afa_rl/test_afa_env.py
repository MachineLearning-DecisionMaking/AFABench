import torch

from afabench.components.initializers.fixed_random_initializer import (
    FixedRandomInitializer,
)
from afabench.components.initializers.manual_initializer import (
    ManualInitializer,
)
from afabench.components.methods.rl.common.afa_env import AFAEnv
from afabench.components.methods.rl.common.dataset_utils import (
    get_afa_dataset_fn,
)
from afabench.components.methods.rl.common.reward_functions import (
    get_fixed_reward_reward_fn,
)
from afabench.components.unmaskers.direct_unmasker import DirectUnmasker
from afabench.components.unmaskers.image_patch_unmasker import (
    ImagePatchUnmasker,
)


def test_initializer_and_unmasker_integration() -> None:
    # Create 2-channel images with simple patterns for easy testing
    # Each sample has 2 channels, each channel is 4x4
    all_features = torch.tensor(
        [
            # Sample 1: channel 0 has sequential values 1-16, channel 1 has values 17-32
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ],  # Channel 0
                [
                    [17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0],
                    [29.0, 30.0, 31.0, 32.0],
                ],  # Channel 1
            ],
            # Sample 2: channel 0 has even values 2-32, channel 1 has odd values 1-31
            [
                [
                    [2.0, 4.0, 6.0, 8.0],
                    [10.0, 12.0, 14.0, 16.0],
                    [18.0, 20.0, 22.0, 24.0],
                    [26.0, 28.0, 30.0, 32.0],
                ],  # Channel 0
                [
                    [1.0, 3.0, 5.0, 7.0],
                    [9.0, 11.0, 13.0, 15.0],
                    [17.0, 19.0, 21.0, 23.0],
                    [25.0, 27.0, 29.0, 31.0],
                ],  # Channel 1
            ],
            # Sample 3: channel 0 has values 0-15, channel 1 has values 16-31
            [
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0, 15.0],
                ],  # Channel 0
                [
                    [16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0],
                    [24.0, 25.0, 26.0, 27.0],
                    [28.0, 29.0, 30.0, 31.0],
                ],  # Channel 1
            ],
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels)
    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((2,)),
        feature_shape=torch.Size((2, 4, 4)),
        n_selections=4,
        n_classes=3,
        hard_budget=2.0,
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=ImagePatchUnmasker(
            image_side_length=4,
            patch_size=2,
            n_channels=2,
        ).unmask,
        seed=123,
    )

    # Get initial state, t = 0
    td = env.reset()
    assert torch.allclose(
        td["feature_mask"], torch.zeros((2, 2, 4, 4), dtype=torch.bool)
    )
    assert torch.allclose(
        td["performed_action_mask"], torch.zeros((2, 5), dtype=torch.bool)
    )
    assert torch.allclose(
        td["allowed_action_mask"], torch.ones((2, 5), dtype=torch.bool)
    )
    assert torch.allclose(
        td["performed_selection_mask"], torch.zeros((2, 4), dtype=torch.bool)
    )
    assert torch.allclose(
        td["masked_features"], torch.zeros((2, 2, 4, 4), dtype=torch.float32)
    )
    # Trust that features and labels are forwarded properly by the environment

    # Pick the second patch for the first sample and the third patch for the second sample
    td["action"] = torch.tensor([2, 3], dtype=torch.int64)

    # t = 1
    td = env.step(td)
    td = td["next"]

    expected_feature_mask_t1 = torch.tensor(
        [
            # Sample 1: second patch unmasked (patch 2: top-right 2x2)
            [
                [
                    [False, False, True, True],
                    [False, False, True, True],
                    [False, False, False, False],
                    [False, False, False, False],
                ],
                [
                    [False, False, True, True],
                    [False, False, True, True],
                    [False, False, False, False],
                    [False, False, False, False],
                ],
            ],
            # Sample 2: third patch unmasked (patch 3: bottom-left 2x2)
            [
                [
                    [False, False, False, False],
                    [False, False, False, False],
                    [True, True, False, False],
                    [True, True, False, False],
                ],
                [
                    [False, False, False, False],
                    [False, False, False, False],
                    [True, True, False, False],
                    [True, True, False, False],
                ],
            ],
        ],
        dtype=torch.bool,
    )
    assert torch.allclose(td["feature_mask"], expected_feature_mask_t1)

    # Check masks - actions 2 and 3 should now be performed and disabled
    expected_performed_action_mask_t1 = torch.tensor(
        [
            [False, False, True, False, False],  # Sample 1: action 2 performed
            [False, False, False, True, False],  # Sample 2: action 3 performed
        ],
        dtype=torch.bool,
    )
    expected_allowed_action_mask_t1 = torch.tensor(
        [
            [True, True, False, True, True],  # Sample 1: action 2 disabled
            [True, True, True, False, True],  # Sample 2: action 3 disabled
        ],
        dtype=torch.bool,
    )
    expected_performed_selection_mask_t1 = torch.tensor(
        [
            [False, True, False, False],  # Sample 1: selection 1 performed
            [False, False, True, False],  # Sample 2: selection 2 performed
        ],
        dtype=torch.bool,
    )
    assert torch.allclose(
        td["performed_action_mask"], expected_performed_action_mask_t1
    )
    assert torch.allclose(
        td["allowed_action_mask"], expected_allowed_action_mask_t1
    )
    assert torch.allclose(
        td["performed_selection_mask"], expected_performed_selection_mask_t1
    )

    # Check masked_features - should show the unmasked values
    expected_masked_features_t1 = torch.zeros(
        (2, 2, 4, 4), dtype=torch.float32
    )
    # Sample 1: patch 2 (top-right) unmasked
    expected_masked_features_t1[0, 0, 0:2, 2:4] = torch.tensor(
        [[3.0, 4.0], [7.0, 8.0]]
    )
    expected_masked_features_t1[0, 1, 0:2, 2:4] = torch.tensor(
        [[19.0, 20.0], [23.0, 24.0]]
    )
    # Sample 2: patch 3 (bottom-left) unmasked
    expected_masked_features_t1[1, 0, 2:4, 0:2] = torch.tensor(
        [[18.0, 20.0], [26.0, 28.0]]
    )
    expected_masked_features_t1[1, 1, 2:4, 0:2] = torch.tensor(
        [[17.0, 19.0], [25.0, 27.0]]
    )
    assert torch.allclose(td["masked_features"], expected_masked_features_t1)

    # Pick the first patch for the first sample and the fourth patch for the second sample
    td["action"] = torch.tensor([1, 4], dtype=torch.int64)

    # t = 2 (reaches hard_budget=2.0)
    td = env.step(td)
    td = td["next"]
    expected_feature_mask_t2 = torch.tensor(
        [
            # Sample 1: patches 1 and 2 unmasked
            [
                [
                    [True, True, True, True],
                    [True, True, True, True],
                    [False, False, False, False],
                    [False, False, False, False],
                ],
                [
                    [True, True, True, True],
                    [True, True, True, True],
                    [False, False, False, False],
                    [False, False, False, False],
                ],
            ],
            # Sample 2: patches 3 and 4 unmasked
            [
                [
                    [False, False, False, False],
                    [False, False, False, False],
                    [True, True, True, True],
                    [True, True, True, True],
                ],
                [
                    [False, False, False, False],
                    [False, False, False, False],
                    [True, True, True, True],
                    [True, True, True, True],
                ],
            ],
        ],
        dtype=torch.bool,
    )
    assert torch.allclose(td["feature_mask"], expected_feature_mask_t2)

    # Check masked_features - should show both patches unmasked
    expected_masked_features_t2 = torch.zeros(
        (2, 2, 4, 4), dtype=torch.float32
    )
    # Sample 1: patches 1 (top-left) and 2 (top-right) unmasked
    expected_masked_features_t2[0, 0, 0:2, 0:2] = torch.tensor(
        [[1.0, 2.0], [5.0, 6.0]]
    )
    expected_masked_features_t2[0, 0, 0:2, 2:4] = torch.tensor(
        [[3.0, 4.0], [7.0, 8.0]]
    )
    expected_masked_features_t2[0, 1, 0:2, 0:2] = torch.tensor(
        [[17.0, 18.0], [21.0, 22.0]]
    )
    expected_masked_features_t2[0, 1, 0:2, 2:4] = torch.tensor(
        [[19.0, 20.0], [23.0, 24.0]]
    )
    # Sample 2: patches 3 (bottom-left) and 4 (bottom-right) unmasked
    expected_masked_features_t2[1, 0, 2:4, 0:2] = torch.tensor(
        [[18.0, 20.0], [26.0, 28.0]]
    )
    expected_masked_features_t2[1, 0, 2:4, 2:4] = torch.tensor(
        [[22.0, 24.0], [30.0, 32.0]]
    )
    expected_masked_features_t2[1, 1, 2:4, 0:2] = torch.tensor(
        [[17.0, 19.0], [25.0, 27.0]]
    )
    expected_masked_features_t2[1, 1, 2:4, 2:4] = torch.tensor(
        [[21.0, 23.0], [29.0, 31.0]]
    )
    assert torch.allclose(td["masked_features"], expected_masked_features_t2)

    assert not td["done"].any(), (
        "Episode should not terminate when budget is reached"
    )

    # t = 3, any further selection would exceed the hard budget
    feature_mask_t2 = td["feature_mask"].clone()
    td["action"] = torch.tensor([3, 1], dtype=torch.int64)
    td = env.step(td)["next"]

    # Forced stop: the proposed selections are not executed
    assert torch.equal(td["feature_mask"], feature_mask_t2)
    assert torch.allclose(td["masked_features"], expected_masked_features_t2)
    assert td["done"].all(), "Episode should be a forced stop"


def test_stop_due_to_hard_budget() -> None:
    """Test a forced stop when a selection would **exceed** the hard budget."""
    # Use simple 1D features for easy testing
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # Sample 1
            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # Sample 2
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0],  # Sample 1 label
            [0, 1],  # Sample 2 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    # Create environment with hard budget of 2
    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((2,)),
        feature_shape=torch.Size((6,)),
        n_selections=6,  # 4 possible selections
        n_classes=2,
        hard_budget=2.0,  # A third selection would exceed the budget
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    # Reset environment
    td = env.reset()

    # First selection - should not terminate
    td["action"] = torch.tensor(
        [1, 2], dtype=torch.int64
    )  # Select features 1 and 2
    td = env.step(td)
    td = td["next"]

    assert not td["done"].any(), (
        "Environment should not terminate after first selection"
    )

    # Second selection - should not terminate due to hard budget
    td["action"] = torch.tensor(
        [3, 4], dtype=torch.int64
    )  # Select features 3 and 4
    td = env.step(td)
    td = td["next"]

    assert not td["done"].any(), (
        "Environment should not terminate when budget is reached, only if it is exceeded."
    )

    # Third selection - would exceed hard budget, so forced stop
    td["action"] = torch.tensor(
        [5, 6], dtype=torch.int64
    )  # Select features 5 and 6
    td = env.step(td)
    td = td["next"]

    assert td["done"].all(), (
        "Environment should force a stop when a selection would exceed the "
        "hard budget"
    )
    assert torch.equal(td["accumulated_cost"], torch.tensor([2.0, 2.0])), (
        "Forced stop should not charge the over-budget selection"
    )


def test_over_budget_selection_is_forced_stop_without_acquisition() -> None:
    """An over-budget selection ends the episode without being executed."""
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0],  # Sample 1
            [4.0, 5.0, 6.0],  # Sample 2
        ]
    )
    all_labels = torch.tensor([[1, 0], [0, 1]])
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=5.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((2,)),
        feature_shape=torch.Size((3,)),
        n_selections=3,
        n_classes=2,
        hard_budget=3.0,
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
        selection_costs=[1.0, 2.0, 3.0],
    )

    td = env.reset()
    td["action"] = torch.tensor([1, 1], dtype=torch.int64)
    td = env.step(td)["next"]
    before = td.clone()

    # Sample 1 proposes selection 2 (cost 3, total 4 > 3): forced stop.
    # Sample 2 proposes selection 1 (cost 2, total 3 <= 3): executed.
    proposed_action = torch.tensor([3, 2], dtype=torch.int64)
    td["action"] = proposed_action.clone()
    td = env.step(td)

    # The proposed action stays in the output exactly as proposed
    assert torch.equal(td["action"], proposed_action)
    after = td["next"]

    # The forced stop leaves the state unchanged and charges no cost
    for key in (
        "features",
        "masked_features",
        "feature_mask",
        "performed_selection_mask",
        "performed_action_mask",
        "allowed_action_mask",
        "accumulated_cost",
    ):
        assert torch.equal(after[key][0], before[key][0]), key
    assert after["done"][0].item(), "Over-budget proposal should be done"
    assert after["reward"][0].item() == 5.0, "Forced stop gets stop reward"

    # The within-budget selection in the same batch is executed normally
    assert torch.equal(
        after["feature_mask"][1], torch.tensor([True, True, False])
    )
    assert after["accumulated_cost"][1].item() == 3.0
    assert not after["done"][1].item()


def test_stop_due_to_no_more_actions() -> None:
    """Test that the environment terminates when all selection actions are exhausted."""
    # Use simple 1D features with only 2 features available for selection
    all_features = torch.tensor(
        [
            [1.0, 2.0],  # Sample 1 - only 2 features
            [3.0, 4.0],  # Sample 2 - only 2 features
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0],  # Sample 1 label
            [0, 1],  # Sample 2 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    # Create environment with only 2 selections possible and high hard budget
    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((2,)),
        feature_shape=torch.Size((2,)),
        n_selections=2,  # Only 2 possible selections
        n_classes=2,
        hard_budget=10,  # High budget, should terminate due to no more actions
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    # Reset environment
    td = env.reset()

    # First selection - should not terminate
    td["action"] = torch.tensor(
        [1, 1], dtype=torch.int64
    )  # Select first feature
    td = env.step(td)
    td = td["next"]

    assert not td["done"].any(), (
        "Environment should not terminate after first selection"
    )

    # Second selection - should terminate because no more selection actions available
    td["action"] = torch.tensor(
        [2, 2], dtype=torch.int64
    )  # Select second feature
    td = env.step(td)
    td = td["next"]

    assert td["done"].all(), (
        "Environment should auto-terminate when no more selection actions are available"
    )


def test_stop_due_to_stop_action() -> None:
    """Test that the environment terminates when stop action (action 0) is chosen."""
    # Use simple 1D features
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # Sample 1
            [5.0, 6.0, 7.0, 8.0],  # Sample 2
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0],  # Sample 1 label
            [0, 1],  # Sample 2 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    # Create environment with high hard budget - should terminate due to stop action
    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((2,)),
        feature_shape=torch.Size((4,)),
        n_selections=4,
        n_classes=2,
        hard_budget=10,  # High budget, should not terminate due to budget
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    # Reset environment
    td = env.reset()

    # Make one selection first - should not terminate
    td["action"] = torch.tensor(
        [1, 2], dtype=torch.int64
    )  # Select features 1 and 2
    td = env.step(td)
    td = td["next"]

    assert not td["done"].any(), (
        "Environment should not terminate after first selection"
    )

    # Choose stop action (action 0) - should terminate
    td["action"] = torch.tensor([0, 0], dtype=torch.int64)  # Stop action
    td = env.step(td)
    td = td["next"]

    assert td["done"].all(), (
        "Environment should terminate when stop action is chosen"
    )


def test_per_sample_termination_hard_budget() -> None:
    """Test that samples terminate independently when reaching hard budget."""
    # Use simple 1D features
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # Sample 1
            [5.0, 6.0, 7.0, 8.0],  # Sample 2
            [9.0, 10.0, 11.0, 12.0],  # Sample 3
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0, 0],  # Sample 1 label
            [0, 1, 0],  # Sample 2 label
            [0, 0, 1],  # Sample 3 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    # Create environment with different hard budgets for samples
    # We'll simulate this by carefully choosing actions
    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((3,)),
        feature_shape=torch.Size((4,)),
        n_selections=4,
        n_classes=3,
        hard_budget=1.99,  # All samples have same hard budget
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    # Reset environment
    td = env.reset()

    # First step: Sample 1 makes selection, samples 2&3 choose stop
    td["action"] = torch.tensor([1, 0, 0], dtype=torch.int64)
    td = env.step(td)
    td = td["next"]

    # Only samples 2 and 3 should be done (they chose stop)
    expected_done = torch.tensor([[False], [True], [True]], dtype=torch.bool)
    assert torch.equal(td["done"], expected_done), (
        f"Expected done={expected_done}, got {td['done']}"
    )

    # Second step: Sample 1 proposes a selection that would exceed the hard
    # budget (forced stop)
    td["action"] = torch.tensor([2, 0, 0], dtype=torch.int64)
    td = env.step(td)
    td = td["next"]

    # Now all samples should be done
    assert td["done"].all(), (
        "All samples should be done - sample 1 was forced to stop by the "
        "hard budget, others already stopped"
    )


def test_per_sample_termination_no_more_actions() -> None:
    """Test that samples terminate independently when exhausting actions."""
    # Use simple 1D features with only 2 features available for selection
    all_features = torch.tensor(
        [
            [1.0, 2.0],  # Sample 1
            [3.0, 4.0],  # Sample 2
            [5.0, 6.0],  # Sample 3
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0, 0],  # Sample 1 label
            [0, 1, 0],  # Sample 2 label
            [0, 0, 1],  # Sample 3 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    # Create environment with only 2 selection actions possible
    # We'll simulate different exhaustion times by having samples stop at different points
    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((3,)),
        feature_shape=torch.Size((2,)),
        n_selections=2,
        n_classes=3,
        hard_budget=10,  # High budget
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    # Reset environment
    td = env.reset()

    # Step 1: Sample 1 makes first selection, sample 2 exhausts all selections, sample 3 makes first selection
    td["action"] = torch.tensor([1, 1, 1], dtype=torch.int64)
    td = env.step(td)
    td = td["next"]

    # No samples should be done yet
    expected_done = torch.tensor([[False], [False], [False]], dtype=torch.bool)
    assert torch.equal(td["done"], expected_done), (
        f"Expected done={expected_done}, got {td['done']}"
    )

    # Step 2: Sample 1 stops, sample 2 makes second selection (should auto-terminate), sample 3 makes second selection (should auto-terminate)
    td["action"] = torch.tensor([0, 2, 2], dtype=torch.int64)
    td = env.step(td)
    td = td["next"]

    # Sample 1 should be done (chose stop), samples 2 and 3 should be done (exhausted all selections)
    expected_done = torch.tensor([[True], [True], [True]], dtype=torch.bool)
    assert torch.equal(td["done"], expected_done), (
        f"Expected done={expected_done}, got {td['done']}"
    )


def test_per_sample_termination_stop_action() -> None:
    """Test that samples terminate independently when choosing stop action."""
    # Use simple 1D features
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # Sample 1
            [5.0, 6.0, 7.0, 8.0],  # Sample 2
            [9.0, 10.0, 11.0, 12.0],  # Sample 3
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0, 0],  # Sample 1 label
            [0, 1, 0],  # Sample 2 label
            [0, 0, 1],  # Sample 3 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((3,)),
        feature_shape=torch.Size((4,)),
        n_selections=4,
        n_classes=3,
        hard_budget=10,  # High budget
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    # Reset environment
    td = env.reset()

    # Step 1: Sample 1 chooses stop, others make selections
    td["action"] = torch.tensor([0, 1, 2], dtype=torch.int64)
    td = env.step(td)
    td = td["next"]

    # Only sample 1 should be done
    expected_done = torch.tensor([[True], [False], [False]], dtype=torch.bool)
    assert torch.equal(td["done"], expected_done), (
        f"Expected done={expected_done}, got {td['done']}"
    )

    # Step 2: All remaining samples make selections
    td["action"] = torch.tensor([0, 3, 1], dtype=torch.int64)
    td = env.step(td)
    td = td["next"]

    # Still only sample 1 should be done
    expected_done = torch.tensor([[True], [False], [False]], dtype=torch.bool)
    assert torch.equal(td["done"], expected_done), (
        f"Expected done={expected_done}, got {td['done']}"
    )

    # Step 3: Sample 2 chooses stop, sample 3 makes selection
    td["action"] = torch.tensor([0, 0, 4], dtype=torch.int64)
    td = env.step(td)
    td = td["next"]

    # Samples 1 and 2 should be done
    expected_done = torch.tensor([[True], [True], [False]], dtype=torch.bool)
    assert torch.equal(td["done"], expected_done), (
        f"Expected done={expected_done}, got {td['done']}"
    )

    # Step 4: Sample 3 chooses stop
    td["action"] = torch.tensor([0, 0, 0], dtype=torch.int64)
    td = env.step(td)
    td = td["next"]

    # All samples should be done
    assert td["done"].all(), "All samples should be done"


def test_invalid_action_handling() -> None:
    """Test environment behavior with invalid actions."""
    # Use simple 1D features
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # Sample 1
            [5.0, 6.0, 7.0, 8.0],  # Sample 2
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0],  # Sample 1 label
            [0, 1],  # Sample 2 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((2,)),
        feature_shape=torch.Size((4,)),
        n_selections=4,
        n_classes=2,
        hard_budget=5,
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    # Reset environment
    td = env.reset()

    # First, perform a valid action to test re-selection
    td["action"] = torch.tensor([1, 2], dtype=torch.int64)
    td = env.step(td)
    td = td["next"]

    # Test trying to select the same action again (should be disallowed)
    td["action"] = torch.tensor([1, 1], dtype=torch.int64)  # Already performed
    td_result = env.step(td)
    td_result = td_result["next"]

    # Environment currently allows re-selection of performed actions
    # This may not be ideal behavior, but we test the current implementation
    # Both samples now have action 1 performed
    assert torch.equal(
        td_result["performed_action_mask"][:, 1],
        torch.tensor([True, True], dtype=torch.bool),
    ), "Action 1 should be performed for both samples (current behavior)"

    # Test out-of-bounds actions (actions > n_selections)
    td = env.reset()
    # Note: We can't easily test out-of-bounds without modifying the environment
    # as PyTorch will raise IndexError. This would need error handling in the env.


def test_mask_consistency() -> None:
    """Test that all masks remain consistent with each other."""
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # Sample 1
            [5.0, 6.0, 7.0, 8.0],  # Sample 2
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0],  # Sample 1 label
            [0, 1],  # Sample 2 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((2,)),
        feature_shape=torch.Size((4,)),
        n_selections=4,
        n_classes=2,
        hard_budget=5,
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    td = env.reset()

    # Perform several actions and check mask consistency at each step
    actions_sequence = [
        torch.tensor([1, 2], dtype=torch.int64),
        torch.tensor([3, 1], dtype=torch.int64),
        torch.tensor([2, 4], dtype=torch.int64),
    ]

    for action in actions_sequence:
        td["action"] = action
        td = env.step(td)
        td = td["next"]

        # Check that performed_action_mask and allowed_action_mask are complementary for selection actions
        for sample_idx in range(td.batch_size[0]):
            for action_idx in range(
                1, env.n_selections + 1
            ):  # Skip stop action (0)
                performed = td["performed_action_mask"][sample_idx, action_idx]
                allowed = td["allowed_action_mask"][sample_idx, action_idx]
                # They should be opposite (if performed, then not allowed)
                assert performed != allowed or not performed, (
                    f"Inconsistent masks for sample {sample_idx}, action {action_idx}: "
                    f"performed={performed}, allowed={allowed}"
                )

        # Check that performed_selection_mask matches performed actions (excluding stop)
        expected_selection_mask = td["performed_action_mask"][
            :, 1:
        ]  # Exclude stop action
        assert torch.equal(
            td["performed_selection_mask"], expected_selection_mask
        ), "Selection mask should match performed actions (excluding stop)"

        # Check that masked_features only has non-zero values where feature_mask is True
        masked_values_exist = td["masked_features"] != 0.0
        feature_mask_positions = td["feature_mask"]

        # Where we have masked values, we should have feature mask True
        # (but not necessarily the other way around, as features could be 0.0)
        valid_masked_positions = masked_values_exist <= feature_mask_positions
        assert valid_masked_positions.all(), (
            "Masked features should only have non-zero values where feature mask is True"
        )


def test_reward_integration() -> None:
    """Test reward calculation with different scenarios."""
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # Sample 1
            [5.0, 6.0, 7.0, 8.0],  # Sample 2
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0],  # Sample 1 label
            [0, 1],  # Sample 2 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    # Test with different reward function
    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=1.0, reward_otherwise=-0.1
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((2,)),
        feature_shape=torch.Size((4,)),
        n_selections=4,
        n_classes=2,
        hard_budget=3,
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    td = env.reset()

    # Test reward for selection action (should be reward_otherwise)
    td["action"] = torch.tensor([1, 2], dtype=torch.int64)
    td = env.step(td)
    td = td["next"]

    expected_reward = torch.tensor([[-0.1], [-0.1]], dtype=torch.float32)
    assert torch.allclose(td["reward"], expected_reward), (
        f"Expected reward {expected_reward}, got {td['reward']}"
    )

    # Test reward for stop action (should be reward_for_stop)
    td["action"] = torch.tensor([0, 0], dtype=torch.int64)
    td = env.step(td)
    td = td["next"]

    expected_reward_stop = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
    assert torch.allclose(td["reward"], expected_reward_stop), (
        f"Expected stop reward {expected_reward_stop}, got {td['reward']}"
    )

    # Check reward shape consistency
    assert td["reward"].shape == (td.batch_size[0], 1), (
        f"Reward shape should be {(td.batch_size[0], 1)}, got {td['reward'].shape}"
    )

    # Check reward device consistency
    assert td["reward"].device == td["features"].device, (
        "Reward device should match other tensors"
    )


def test_state_immutability() -> None:
    """Test that input tensordicts are not modified."""
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # Sample 1
            [5.0, 6.0, 7.0, 8.0],  # Sample 2
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0],  # Sample 1 label
            [0, 1],  # Sample 2 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((2,)),
        feature_shape=torch.Size((4,)),
        n_selections=4,
        n_classes=2,
        hard_budget=5,
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    td = env.reset()

    # Create deep copies of all tensors to check immutability
    original_tensors = {}
    for key, value in td.items():
        original_tensors[key] = value.clone()

    # Perform an action
    td["action"] = torch.tensor([1, 2], dtype=torch.int64)

    # Store the action for comparison
    original_action = td["action"].clone()

    # Step the environment
    result_td = env.step(td)

    # Check that original tensordict was not modified
    for key, original_value in original_tensors.items():
        if key != "action":  # We explicitly set action, so skip it
            current_value = td[key]
            assert torch.equal(current_value, original_value), (
                f"Original tensordict key '{key}' was modified during step"
            )

    # Check that the action we set is still the same
    assert torch.equal(td["action"], original_action), (
        "Original action tensor was modified"
    )

    # The result should be a new tensordict
    result_state = result_td["next"]
    assert result_state is not td, "Result should be a new tensordict"


def test_different_batch_sizes() -> None:
    """Test environment with various batch sizes."""
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # Sample 1
            [5.0, 6.0, 7.0, 8.0],  # Sample 2
            [9.0, 10.0, 11.0, 12.0],  # Sample 3
            [13.0, 14.0, 15.0, 16.0],  # Sample 4
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0, 0, 0],  # Sample 1 label
            [0, 1, 0, 0],  # Sample 2 label
            [0, 0, 1, 0],  # Sample 3 label
            [0, 0, 0, 1],  # Sample 4 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    # Test with batch size 1
    env_single = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((1,)),
        feature_shape=torch.Size((4,)),
        n_selections=4,
        n_classes=4,
        hard_budget=2,
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    td_single = env_single.reset()
    assert td_single.batch_size == torch.Size([1]), (
        "Single batch size should be [1]"
    )

    td_single["action"] = torch.tensor([1], dtype=torch.int64)
    td_single = env_single.step(td_single)
    td_single = td_single["next"]

    assert td_single["done"].shape == (
        1,
        1,
    ), "Done shape should match batch size"
    assert td_single["reward"].shape == (
        1,
        1,
    ), "Reward shape should match batch size"

    # Test with larger batch size (4)
    env_large = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((4,)),
        feature_shape=torch.Size((4,)),
        n_selections=4,
        n_classes=4,
        hard_budget=2,
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    td_large = env_large.reset()
    assert td_large.batch_size == torch.Size([4]), (
        "Large batch size should be [4]"
    )

    td_large["action"] = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    td_large = env_large.step(td_large)
    td_large = td_large["next"]

    assert td_large["done"].shape == (
        4,
        1,
    ), "Done shape should match batch size"
    assert td_large["reward"].shape == (
        4,
        1,
    ), "Reward shape should match batch size"
    assert td_large["feature_mask"].shape == (
        4,
        4,
    ), "Feature mask should match batch and features"


def test_environment_reset_behavior() -> None:
    """Test environment reset behavior and state consistency."""
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # Sample 1
            [5.0, 6.0, 7.0, 8.0],  # Sample 2
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0],  # Sample 1 label
            [0, 1],  # Sample 2 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((2,)),
        feature_shape=torch.Size((4,)),
        n_selections=4,
        n_classes=2,
        hard_budget=5,
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    # First reset
    td1 = env.reset()

    # Verify initial state
    assert torch.equal(
        td1["feature_mask"], torch.zeros((2, 4), dtype=torch.bool)
    ), "Initial feature mask should be all False"
    assert torch.equal(
        td1["performed_action_mask"], torch.zeros((2, 5), dtype=torch.bool)
    ), "Initial performed action mask should be all False"
    assert torch.equal(
        td1["allowed_action_mask"], torch.ones((2, 5), dtype=torch.bool)
    ), "Initial allowed action mask should be all True"
    assert torch.equal(
        td1["masked_features"], torch.zeros((2, 4), dtype=torch.float32)
    ), "Initial masked features should be all zero"

    # Perform some actions
    td1["action"] = torch.tensor([1, 2], dtype=torch.int64)
    td1 = env.step(td1)
    td1 = td1["next"]

    td1["action"] = torch.tensor([3, 1], dtype=torch.int64)
    td1 = env.step(td1)
    td1 = td1["next"]

    # Reset again and verify state is restored
    td2 = env.reset()

    # Second reset should give same initial state as first
    assert torch.equal(td1["features"], td2["features"]), (
        "Features should be the same after reset"
    )
    assert torch.equal(td1["label"], td2["label"]), (
        "Labels should be the same after reset"
    )
    assert torch.equal(
        td2["feature_mask"], torch.zeros((2, 4), dtype=torch.bool)
    ), "Feature mask should be reset to all False"
    assert torch.equal(
        td2["performed_action_mask"], torch.zeros((2, 5), dtype=torch.bool)
    ), "Performed action mask should be reset"
    assert torch.equal(
        td2["allowed_action_mask"], torch.ones((2, 5), dtype=torch.bool)
    ), "Allowed action mask should be reset"

    # Test multiple consecutive resets
    td3 = env.reset()
    td4 = env.reset()

    assert torch.equal(td3["features"], td4["features"]), (
        "Multiple resets should be consistent"
    )
    assert torch.equal(td3["feature_mask"], td4["feature_mask"]), (
        "Reset state should be consistent"
    )


def test_tensordict_structure_validation() -> None:
    """Test that TensorDict structure and types are correct."""
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # Sample 1
            [5.0, 6.0, 7.0, 8.0],  # Sample 2
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0],  # Sample 1 label
            [0, 1],  # Sample 2 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((2,)),
        feature_shape=torch.Size((4,)),
        n_selections=4,
        n_classes=2,
        hard_budget=5,
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    td = env.reset()

    # Check that all expected keys are present
    expected_keys = {
        "features",
        "label",
        "feature_mask",
        "performed_action_mask",
        "allowed_action_mask",
        "performed_selection_mask",
        "masked_features",
    }
    assert set(td.keys()) >= expected_keys, (
        f"Missing keys: {expected_keys - set(td.keys())}"
    )

    # Check tensor dtypes
    assert td["features"].dtype == torch.float32, "Features should be float32"
    assert td["label"].dtype == torch.int64, "Labels should be int64"
    assert td["feature_mask"].dtype == torch.bool, (
        "Feature mask should be bool"
    )
    assert td["performed_action_mask"].dtype == torch.bool, (
        "Performed action mask should be bool"
    )
    assert td["allowed_action_mask"].dtype == torch.bool, (
        "Allowed action mask should be bool"
    )
    assert td["performed_selection_mask"].dtype == torch.bool, (
        "Performed selection mask should be bool"
    )
    assert td["masked_features"].dtype == torch.float32, (
        "Masked features should be float32"
    )

    # Check tensor shapes
    batch_size = td.batch_size[0]
    assert td["features"].shape == (
        batch_size,
        4,
    ), f"Features shape should be ({batch_size}, 4)"
    assert td["label"].shape == (
        batch_size,
        2,
    ), f"Label shape should be ({batch_size}, 2)"
    assert td["feature_mask"].shape == (
        batch_size,
        4,
    ), f"Feature mask shape should be ({batch_size}, 4)"
    assert td["performed_action_mask"].shape == (
        batch_size,
        5,
    ), f"Performed action mask shape should be ({batch_size}, 5)"
    assert td["allowed_action_mask"].shape == (
        batch_size,
        5,
    ), f"Allowed action mask shape should be ({batch_size}, 5)"
    assert td["performed_selection_mask"].shape == (
        batch_size,
        4,
    ), f"Performed selection mask shape should be ({batch_size}, 4)"
    assert td["masked_features"].shape == (
        batch_size,
        4,
    ), f"Masked features shape should be ({batch_size}, 4)"

    # Perform a step and check result structure
    td["action"] = torch.tensor([1, 2], dtype=torch.int64)
    result_td = env.step(td)

    # Check that result has "next" key
    assert "next" in result_td, "Step result should contain 'next' key"

    next_td = result_td["next"]

    # Check that next state has additional keys
    additional_keys = {"done", "reward"}
    assert set(next_td.keys()) >= expected_keys | additional_keys, (
        f"Next state missing keys: {(expected_keys | additional_keys) - set(next_td.keys())}"
    )

    # Check new tensor types and shapes
    assert next_td["done"].dtype == torch.bool, "Done should be bool"
    assert next_td["reward"].dtype == torch.float32, "Reward should be float32"
    assert next_td["done"].shape == (
        batch_size,
        1,
    ), f"Done shape should be ({batch_size}, 1)"
    assert next_td["reward"].shape == (
        batch_size,
        1,
    ), f"Reward shape should be ({batch_size}, 1)"

    # Check device consistency
    device = td["features"].device
    for key, tensor in next_td.items():
        if isinstance(tensor, torch.Tensor):
            assert tensor.device == device, (
                f"Tensor '{key}' should be on device {device}"
            )


def test_allow_stop_action_true() -> None:
    """Test that when allow_stop_action=True, the stop action (index 0) is allowed."""
    device = torch.device("cpu")
    batch_size = torch.Size((2,))
    feature_shape = torch.Size((2, 4, 4))
    n_selections = 8
    n_classes = 10

    # Create dummy data
    features = torch.randn(batch_size + feature_shape)
    labels = torch.randn(batch_size + (n_classes,))

    def dummy_dataset_fn(
        batch_size: torch.Size,
        *,
        move_on: bool = True,  # noqa: ARG001
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return features[: batch_size[0]], labels[: batch_size[0]]

    # Create environment with allow_stop_action=True (hard_budget=None)
    env = AFAEnv(
        dataset_fn=dummy_dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=device,
        batch_size=batch_size,
        feature_shape=feature_shape,
        n_selections=n_selections,
        n_classes=n_classes,
        hard_budget=None,  # This should set allow_stop_action=True
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    # Reset environment
    td = env.reset()

    # Check that stop action (index 0) is allowed
    assert td["allowed_action_mask"][:, 0].all(), (
        "Stop action should be allowed when allow_stop_action=True"
    )

    # Take a non-stop action
    action_td = td.clone()
    action_td["action"] = torch.tensor(
        [1, 2], dtype=torch.int64
    )  # Non-stop actions
    td_next = env.step(action_td)

    # Check that stop action is still allowed after step
    assert td_next["allowed_action_mask"][:, 0].all(), (
        "Stop action should remain allowed after step when allow_stop_action=True"
    )


def test_force_hard_budget_false_default() -> None:
    """Test that when force_hard_budget=False (default) and hard_budget is set, the stop action (index 0) is still allowed."""
    device = torch.device("cpu")
    batch_size = torch.Size((2,))
    feature_shape = torch.Size((2, 4, 4))
    n_selections = 8
    n_classes = 10

    # Create dummy data
    features = torch.randn(batch_size + feature_shape)
    labels = torch.randn(batch_size + (n_classes,))

    def dummy_dataset_fn(
        batch_size: torch.Size,
        *,
        move_on: bool = True,  # noqa: ARG001
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return features[: batch_size[0]], labels[: batch_size[0]]

    # Create environment with allow_stop_action=True (hard_budget set but force_hard_budget=False by default)
    env = AFAEnv(
        dataset_fn=dummy_dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=device,
        batch_size=batch_size,
        feature_shape=feature_shape,
        n_selections=n_selections,
        n_classes=n_classes,
        hard_budget=5,  # With force_hard_budget=False (default), stop action should be allowed
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    # Reset environment
    td = env.reset()

    # Check that stop action (index 0) is allowed
    assert td["allowed_action_mask"][:, 0].all(), (
        "Stop action should be allowed when force_hard_budget=False (default)"
    )

    # Take a non-stop action
    action_td = td.clone()
    action_td["action"] = torch.tensor(
        [1, 2], dtype=torch.int64
    )  # Non-stop actions
    td_next = env.step(action_td)

    # Check that stop action is still allowed after step
    assert td_next["allowed_action_mask"][:, 0].all(), (
        "Stop action should remain allowed after step when force_hard_budget=False"
    )


def test_force_hard_budget_true() -> None:
    """Test that when force_hard_budget=True and hard_budget is set, the stop action (index 0) is not allowed."""
    device = torch.device("cpu")
    batch_size = torch.Size((2,))
    feature_shape = torch.Size((2, 4, 4))
    n_selections = 8
    n_classes = 10

    # Create dummy data
    features = torch.randn(batch_size + feature_shape)
    labels = torch.randn(batch_size + (n_classes,))

    def dummy_dataset_fn(
        batch_size: torch.Size,
        *,
        move_on: bool = True,  # noqa: ARG001
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return features[: batch_size[0]], labels[: batch_size[0]]

    # Create environment with force_hard_budget=True
    env = AFAEnv(
        dataset_fn=dummy_dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=device,
        batch_size=batch_size,
        feature_shape=feature_shape,
        n_selections=n_selections,
        n_classes=n_classes,
        hard_budget=5,
        force_hard_budget=True,  # This should set allow_stop_action=False
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    # Reset environment
    td = env.reset()

    # Check that stop action (index 0) is not allowed
    assert not td["allowed_action_mask"][:, 0].any(), (
        "Stop action should not be allowed when force_hard_budget=True"
    )

    # Take a non-stop action
    action_td = td.clone()
    action_td["action"] = torch.tensor(
        [1, 2], dtype=torch.int64
    )  # Non-stop actions
    td_next = env.step(action_td)

    # Check that stop action is still not allowed after step
    assert not td_next["allowed_action_mask"][:, 0].any(), (
        "Stop action should remain disallowed after step when force_hard_budget=True"
    )


def test_force_hard_budget_true_consistency_across_steps() -> None:
    """Test that force_hard_budget=True behavior is consistent across multiple steps."""
    device = torch.device("cpu")
    batch_size = torch.Size((1,))
    feature_shape = torch.Size((2, 4, 4))
    n_selections = 8
    n_classes = 10

    # Create dummy data
    features = torch.randn(batch_size + feature_shape)
    labels = torch.randn(batch_size + (n_classes,))

    def dummy_dataset_fn(
        batch_size: torch.Size,
        *,
        move_on: bool = True,  # noqa: ARG001
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return features[: batch_size[0]], labels[: batch_size[0]]

    # Test with force_hard_budget=True
    env = AFAEnv(
        dataset_fn=dummy_dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=device,
        batch_size=batch_size,
        feature_shape=feature_shape,
        n_selections=n_selections,
        n_classes=n_classes,
        hard_budget=3,
        force_hard_budget=True,  # This should set allow_stop_action=False
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    td = env.reset()

    # Take multiple steps and ensure stop action stays disabled
    for i in range(3):  # Take 3 steps (up to hard_budget)
        # Verify stop action is disabled
        assert not td["allowed_action_mask"][:, 0].any(), (
            f"Stop action should be disabled at step {i}"
        )

        # Find an available non-stop action
        available_actions = torch.nonzero(
            td["allowed_action_mask"][0, 1:], as_tuple=False
        )
        if len(available_actions) == 0:
            break  # No more actions available

        action = (
            available_actions[0].item() + 1
        )  # +1 because we're looking at indices 1:
        action_td = td.clone()
        action_td["action"] = torch.tensor([action], dtype=torch.int64)
        td = env.step(action_td)


def test_force_hard_budget_false_consistency_across_steps() -> None:
    """Test that force_hard_budget=False allows stop action even with hard_budget set."""
    device = torch.device("cpu")
    batch_size = torch.Size((1,))
    feature_shape = torch.Size((2, 4, 4))
    n_selections = 8
    n_classes = 10

    # Create dummy data
    features = torch.randn(batch_size + feature_shape)
    labels = torch.randn(batch_size + (n_classes,))

    def dummy_dataset_fn(
        batch_size: torch.Size,
        *,
        move_on: bool = True,  # noqa: ARG001
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return features[: batch_size[0]], labels[: batch_size[0]]

    # Test with force_hard_budget=False (default)
    env = AFAEnv(
        dataset_fn=dummy_dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=device,
        batch_size=batch_size,
        feature_shape=feature_shape,
        n_selections=n_selections,
        n_classes=n_classes,
        hard_budget=3,
        force_hard_budget=False,  # Stop action should be allowed
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    td = env.reset()

    # Take multiple steps and ensure stop action stays enabled
    for i in range(3):  # Take 3 steps (up to hard_budget)
        # Verify stop action is enabled
        assert td["allowed_action_mask"][:, 0].all(), (
            f"Stop action should be enabled at step {i} when force_hard_budget=False"
        )

        # Find an available non-stop action
        available_actions = torch.nonzero(
            td["allowed_action_mask"][0, 1:], as_tuple=False
        )
        if len(available_actions) == 0:
            break  # No more actions available

        action = (
            available_actions[0].item() + 1
        )  # +1 because we're looking at indices 1:
        action_td = td.clone()
        action_td["action"] = torch.tensor([action], dtype=torch.int64)
        td = env.step(action_td)


def test_force_hard_budget_with_different_settings() -> None:
    """Test force_hard_budget behavior with different settings."""
    device = torch.device("cpu")
    batch_size = torch.Size((1,))
    feature_shape = torch.Size((2, 4, 4))
    n_selections = 8
    n_classes = 10

    # Create dummy data
    features = torch.randn(batch_size + feature_shape)
    labels = torch.randn(batch_size + (n_classes,))

    def dummy_dataset_fn(
        batch_size: torch.Size,
        *,
        move_on: bool = True,  # noqa: ARG001
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return features[: batch_size[0]], labels[: batch_size[0]]

    # Test case 1: hard_budget=None should allow stop action
    env1 = AFAEnv(
        dataset_fn=dummy_dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=device,
        batch_size=batch_size,
        feature_shape=feature_shape,
        n_selections=n_selections,
        n_classes=n_classes,
        hard_budget=None,
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    assert env1.allow_stop_action, (
        "allow_stop_action should be True when hard_budget=None"
    )
    td1 = env1.reset()
    assert td1["allowed_action_mask"][:, 0].all(), (
        "Stop action should be allowed when hard_budget=None"
    )

    # Test case 2: hard_budget=1 with force_hard_budget=False should allow stop action
    env2 = AFAEnv(
        dataset_fn=dummy_dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=device,
        batch_size=batch_size,
        feature_shape=feature_shape,
        n_selections=n_selections,
        n_classes=n_classes,
        hard_budget=1,
        force_hard_budget=False,  # Default value
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    assert env2.allow_stop_action, (
        "allow_stop_action should be True when force_hard_budget=False"
    )
    td2 = env2.reset()
    assert td2["allowed_action_mask"][:, 0].all(), (
        "Stop action should be allowed when force_hard_budget=False"
    )

    # Test case 3: hard_budget=1 with force_hard_budget=True should not allow stop action
    env3 = AFAEnv(
        dataset_fn=dummy_dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=device,
        batch_size=batch_size,
        feature_shape=feature_shape,
        n_selections=n_selections,
        n_classes=n_classes,
        hard_budget=1,
        force_hard_budget=True,
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    assert not env3.allow_stop_action, (
        "allow_stop_action should be False when force_hard_budget=True"
    )
    td3 = env3.reset()
    assert not td3["allowed_action_mask"][:, 0].any(), (
        "Stop action should not be allowed when force_hard_budget=True"
    )


def test_force_hard_budget_comprehensive() -> None:
    """Comprehensive test demonstrating all force_hard_budget behaviors."""
    device = torch.device("cpu")
    batch_size = torch.Size((1,))
    feature_shape = torch.Size((2, 4, 4))
    n_selections = 4
    n_classes = 3

    # Create dummy data
    features = torch.randn(batch_size + feature_shape)
    labels = torch.randn(batch_size + (n_classes,))

    def dummy_dataset_fn(
        batch_size: torch.Size,
        *,
        move_on: bool = True,  # noqa: ARG001
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return features[: batch_size[0]], labels[: batch_size[0]]

    # Test 1: hard_budget=None - stop action always allowed
    env1 = AFAEnv(
        dataset_fn=dummy_dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=device,
        batch_size=batch_size,
        feature_shape=feature_shape,
        n_selections=n_selections,
        n_classes=n_classes,
        hard_budget=None,
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )
    assert env1.allow_stop_action
    td1 = env1.reset()
    assert td1["allowed_action_mask"][:, 0].all()

    # Test 2: hard_budget=2, force_hard_budget=False (default) - stop action allowed
    env2 = AFAEnv(
        dataset_fn=dummy_dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=device,
        batch_size=batch_size,
        feature_shape=feature_shape,
        n_selections=n_selections,
        n_classes=n_classes,
        hard_budget=2,
        force_hard_budget=False,
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )
    assert env2.allow_stop_action
    td2 = env2.reset()
    assert td2["allowed_action_mask"][:, 0].all()

    # Test 3: hard_budget=2, force_hard_budget=True - stop action disabled
    env3 = AFAEnv(
        dataset_fn=dummy_dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=device,
        batch_size=batch_size,
        feature_shape=feature_shape,
        n_selections=n_selections,
        n_classes=n_classes,
        hard_budget=2,
        force_hard_budget=True,
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )
    assert not env3.allow_stop_action
    td3 = env3.reset()
    assert not td3["allowed_action_mask"][:, 0].any()


def test_initializer_application() -> None:
    """Test that initializer is correctly applied to set initial features."""
    # Use simple 1D features for easy testing
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],  # Sample 1
            [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],  # Sample 2
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0],  # Sample 1 label
            [0, 1],  # Sample 2 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    # Use ManualInitializer to initialize specific features
    # Let's initialize features at flat indices 1, 3, 5 (which correspond to positions 1, 3, 5 in the 8-element vector)
    manual_init = ManualInitializer(flat_feature_indices=[1, 3, 5])

    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((2,)),
        feature_shape=torch.Size((8,)),
        n_selections=8,
        n_classes=2,
        hard_budget=5,
        initialize_fn=manual_init.initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    # Reset environment to apply initialization
    td = env.reset()

    # Check that feature_mask has the expected pattern
    # Features at indices 1, 3, 5 should be True (unmasked), others should be False
    expected_feature_mask = torch.tensor(
        [
            [False, True, False, True, False, True, False, False],  # Sample 1
            [False, True, False, True, False, True, False, False],  # Sample 2
        ],
        dtype=torch.bool,
    )
    assert torch.equal(td["feature_mask"], expected_feature_mask), (
        f"Expected feature_mask={expected_feature_mask}, got {td['feature_mask']}"
    )

    # Check that masked_features shows the correct initial values
    expected_masked_features = torch.zeros((2, 8), dtype=torch.float32)
    # Sample 1: features at indices 1, 3, 5 should be 2.0, 4.0, 6.0
    expected_masked_features[0, [1, 3, 5]] = torch.tensor([2.0, 4.0, 6.0])
    # Sample 2: features at indices 1, 3, 5 should be 10.0, 12.0, 14.0
    expected_masked_features[1, [1, 3, 5]] = torch.tensor([10.0, 12.0, 14.0])

    assert torch.allclose(td["masked_features"], expected_masked_features), (
        f"Expected masked_features={expected_masked_features}, got {td['masked_features']}"
    )

    # Verify that performed_selection_mask reflects the initialized features
    # Since ManualInitializer directly sets features (not through selections),
    # this depends on how the environment handles initial state
    # The performed_selection_mask should remain all False since no actual selections were made
    expected_performed_selection_mask = torch.zeros((2, 8), dtype=torch.bool)
    assert torch.equal(
        td["performed_selection_mask"], expected_performed_selection_mask
    ), "No selections should be marked as performed during initialization"

    # Test that we can still make selections after initialization
    td["action"] = torch.tensor(
        [2, 4], dtype=torch.int64
    )  # Select features 2 and 4 (1-based)
    td = env.step(td)
    td = td["next"]

    # Check that the new features are now also unmasked
    expected_feature_mask_after_step = expected_feature_mask.clone()
    expected_feature_mask_after_step[0, 1] = (
        True  # Feature at index 1 (action 2-1)
    )
    expected_feature_mask_after_step[1, 3] = (
        True  # Feature at index 3 (action 4-1)
    )

    assert torch.equal(td["feature_mask"], expected_feature_mask_after_step), (
        "Feature mask should include both initialized and selected features"
    )

    # Check that performed_selection_mask now reflects the actual selections made
    expected_performed_selection_mask_after_step = torch.zeros(
        (2, 8), dtype=torch.bool
    )
    expected_performed_selection_mask_after_step[0, 1] = (
        True  # Selection 1 (action 2-1)
    )
    expected_performed_selection_mask_after_step[1, 3] = (
        True  # Selection 3 (action 4-1)
    )

    assert torch.equal(
        td["performed_selection_mask"],
        expected_performed_selection_mask_after_step,
    ), "Only actual selections should be marked in performed_selection_mask"


def test_accumulated_cost_tracked() -> None:
    """Test that the environment keeps track of accumulated costs."""
    # Use simple 1D features for easy testing
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # Sample 1
            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # Sample 2
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0],  # Sample 1 label
            [0, 1],  # Sample 2 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((2,)),
        feature_shape=torch.Size((6,)),
        n_selections=6,  # 4 possible selections
        n_classes=2,
        hard_budget=20.0,  # Large, should not terminate
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
        selection_costs=[1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
    )

    # Reset environment
    td = env.reset()

    td["action"] = torch.tensor(
        [1, 2], dtype=torch.int64
    )  # Select features 1 and 2
    td = env.step(td)
    td = td["next"]

    assert torch.allclose(td["accumulated_cost"], torch.tensor([1.1, 2.1]))

    td["action"] = torch.tensor(
        [3, 4], dtype=torch.int64
    )  # Select features 3 and 4
    td = env.step(td)
    td = td["next"]

    assert torch.allclose(
        td["accumulated_cost"], torch.tensor([1.1 + 3.1, 2.1 + 4.1])
    )

    td["action"] = torch.tensor(
        [5, 6], dtype=torch.int64
    )  # Select features 4 and 5
    td = env.step(td)
    td = td["next"]

    assert torch.allclose(
        td["accumulated_cost"],
        torch.tensor([1.1 + 3.1 + 5.1, 2.1 + 4.1 + 6.1]),
    )


def test_partial_reset_only_restarts_reset_episodes() -> None:
    """A partial reset starts fresh episodes only where `_reset` is set."""
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # Sample 1
            [5.0, 6.0, 7.0, 8.0],  # Sample 2
            [9.0, 10.0, 11.0, 12.0],  # Sample 3
            [13.0, 14.0, 15.0, 16.0],  # Sample 4
        ]
    )
    all_labels = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]])
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((2,)),
        feature_shape=torch.Size((4,)),
        n_selections=4,
        n_classes=2,
        hard_budget=10.0,
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
        selection_costs=[1.5, 2.5, 3.5, 4.5],
    )

    td = env.reset()
    td["action"] = torch.tensor([1, 2], dtype=torch.int64)
    td = env.step(td)["next"]
    ongoing = td.clone()

    # Reset only the second episode
    td["_reset"] = torch.tensor([[False], [True]])
    td = env.reset(td)

    # The first episode continues untouched
    for key in (
        "features",
        "label",
        "feature_mask",
        "masked_features",
        "performed_selection_mask",
        "performed_action_mask",
        "allowed_action_mask",
        "accumulated_cost",
    ):
        assert torch.equal(td[key][0], ongoing[key][0]), key
    assert td["accumulated_cost"][0] == 1.5

    # The second episode restarts on the next training instance
    assert torch.equal(td["features"][1], all_features[2])
    assert torch.equal(td["label"][1], all_labels[2])
    assert not td["feature_mask"][1].any()
    assert not td["masked_features"][1].any()
    assert not td["performed_selection_mask"][1].any()
    assert not td["performed_action_mask"][1].any()
    assert td["allowed_action_mask"][1].all()
    assert td["accumulated_cost"][1] == 0.0
    assert not td["done"].any()

    # Only one training instance was consumed by the partial reset
    td = env.reset()
    assert torch.equal(td["features"], all_features[[3, 0]])
