# Add a method

Implement a method, register it, and run it through the pipeline.

## 1. Implement the method class

Implement `AFAMethod` from `src/afabench/core/types.py`, using
`AFAMethodBase` from `afabench.components.methods.common`:

- Call `super().__init__(device)`.
- Declare `has_builtin_classifier`.
- Implement `act` and `predict`, decorated with `on_method_device`.
- Implement `save` and `load`, accepting a device on load.
- Override `cost_param` and `set_cost_param` for a soft-budget parameter,
  and `set_seed` for a method-owned generator.
- Extend `to` for modules in containers, calling `super().to(device)` first.

Use `src/afabench/components/methods/dummy/with_classifier.py` as a complete
example. Use `save_classifier` and `load_classifier` for a bundled classifier.
Register each new method or classifier class in `REGISTERED_CLASSES` in
`src/afabench/core/registry.py`.
[Bundle format](../reference/bundle_format.md).

## 2. Write the training script and config

Read `TrainingContract` in `src/afabench/fit/contract.py`. Accept all its
`key=value` CLI arguments and save a loadable method bundle to `save_path`.

Copy the structure of `scripts/train_method/random_dummy.py` and
`src/afabench/components/methods/dummy/{config,train}.py`:

1. Define a frozen, keyword-only config dataclass inheriting `TrainingContract`.
2. Give every method hyperparameter a default.
3. Register it with `store_contract_config`.
4. Write a training function taking the config and `FitInputs`, returning
   the trained method.
5. Wrap it in the entry-point script with `load_inputs`, `fit_run` and `save_result`:

```python
@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/train_method/example",
    config_name="config",
)
def main(cfg: ExampleTrainConfig) -> None:
    cfg = cast("ExampleTrainConfig", OmegaConf.to_object(cfg))
    inputs = load_inputs(cfg)
    with fit_run(cfg, tags=["example"], inputs=inputs):
        afa_method = train_example(cfg, inputs)
        save_result(afa_method, cfg, inputs=inputs)
```

Import `load_inputs` from `afabench.fit.inputs`, and `fit_run` and
`save_result` from `afabench.fit.run`. Pass the same `inputs` to both lifecycle helpers.
Use `inputs.train_dataset()`, `val_dataset()`, `initializer()`, `unmasker()`,
`classifier(expected_type)` and `pretrained_model(expected_type)` as needed.
Use the logger yielded by `fit_run` for training metrics.

Apply smoke-test limits before loading inputs, using
`afabench.fit.smoke_test.limit`, for example `limit(cfg, nepochs=1, patience=1)`.

Create `conf/scripts/train_method/example/config.yaml`:

```yaml
hydra:
  searchpath:
    - file://conf
    - file://conf/global

defaults:
  - train_example  # the name registered with store_contract_config
  - hydra: custom
  - initializer: ???
  - unmasker: ???
  - dataset_key: ???
  - _self_
  - optional dataset_override@_global_: ${dataset_key}
  - override hydra/job_logging: output_dir_colorlog
  - override hydra/hydra_logging: colorlog
  - override hydra/launcher: custom_slurm
```

Follow [`conf/components/README.md`](../../conf/components/README.md)
when composing Hydra groups.

### Use a script without Hydra or the helpers

If writing the entry point yourself:

- Parse the contract's `key=value` arguments from `sys.argv`.
- Handle seeding, logging and smoke-test limits yourself.
- Save to `save_path` with `afabench.core.bundle_system.bundle.save_bundle`.
- Supply a provenance record using `capture_provenance`, `bundle_input` and
  `shared_bundle_dataset_identity`; follow `afabench.fit.run.save_result`.
  Include the contract seed, method name and input bundles.

## 3. Add pretraining if needed

Create `scripts/pretrain_model/<name>.py` and its config under
`conf/scripts/pretrain_model/<name>/`. Use a separate config inheriting
`PretrainingContract`; follow the training script's lifecycle pattern.

For supervised Lightning pretraining, use `SupervisedPretrainingConfig` and
`supervised_pretraining_main`; copy `scripts/pretrain_model/jafa.py`.

Add the named model to your profile's pretrain-mapping file:

```yaml
pretrain_mapping:
  example_model:
    pretrain_script_name: "example"
    pretrain_params: []
```

## 4. Add dataset-specific hyperparameters

Put only differing values in
`conf/scripts/train_method/example/dataset_override/<dataset_key>.yaml`.
Do the same for pretraining where needed.

- Keep contract fields such as `seed`, `device` and budgets out of overrides.
- Share repeated hyperparameters through family files included with `defaults:`.
- For tabular and image networks, follow the `architecture_kind` pattern in
  DIME or GDFS rather than selecting a new Hydra architecture group in an override.

## 5. Register the method with your pipeline profile

Update the files selected by your profile under `workflow/conf/`:

```yaml
# methods/<variant>.yaml
methods: [example_method]  # append to the existing list
```

```yaml
# method_options/<variant>.yaml
method_options:
  example_method:
    train_script_name: "example"
    pretrained_model_name: "example_model"  # omit without pretraining
    eval_batch_size:
      default: 128
    soft_budget_datasets: [cube, miniboone]
```

```yaml
# soft_budget_params/<variant>.yaml
soft_budget_params:
  example_method:
    default: []
    cube: [[0.1, null], [0.2, null]]
    miniboone: [[0.1, null], [0.2, null]]
```

Keep a `soft_budget_params` entry even for hard-budget-only methods. Choose
values and opt-ins appropriate to your method.
[Method options](../reference/pipeline_configuration.md#method-option-keys-and-validation).

Optionally add the name to `method_sets/<variant>.yaml`.
Add its display name, policy family and any new family colours in
`conf/scripts/plotting/common/default.yaml`.

## 6. Test and run the method

Run the method's contract-conformance case:

```shell
uv run pytest test/scripts/test_training_contract_conformance.py -m pipeline -k example_method
```

Check config completeness and dataset overrides:

```shell
uv run pytest test/workflow/test_pipeline_script_configs_complete.py \
    test/workflow/test_dataset_overrides_omit_contract_fields.py -m workflow
```

Run a CPU smoke test, adding your profile name if it is not `all`:

```shell
uv run workflow/run.py --jobs 8 \
    "methods=[example_method]" "datasets=[cube]" \
    "dataset_realization_indices=[0]" smoke_test=true \
    execution_file=workflow/profiles/execution/cpu.yaml
```

Run `uv run just qa` before committing. For a production run, remove `smoke_test=true`
and select the intended datasets, realizations and
[execution settings](customize_pipeline.md#choose-cpus-or-gpus).
