# Bundle format

Every object that needs to be saved/loaded follows the same format.

---

## Usage

Use `afabench.core.bundle_system.bundle.save_bundle()` to save bundles and `afabench.core.bundle_system.bundle.load_bundle()` to load bundles. `load_bundle(path, *, expect=None, **kwargs)` reads the manifest once and returns a `LoadedBundle` with the loaded object (`obj`), the typed manifest (`manifest`: a `Manifest` with `bundle_version`, `class_name`, `class_version`, `metadata`, `provenance` and `content_hash`) and the path as given; `as_input(role)` describes it as an input of another artifact, with the path recorded as given. With `expect`, an object that is not an instance of that class raises `TypeError`; other keyword arguments, such as `device`, go to the object's `load` method. `load_bundle_from_manifest(path, manifest, ...)` loads from a manifest already read with `Manifest.read(path)`, so that it is not read again. `save_bundle` requires the bundle's [provenance record](#provenance) as a keyword-only argument; `read_manifest()` reads a manifest without loading the object, `bundle_provenance()` reads a bundle's record, `shared_bundle_dataset_identity()` returns the dataset identity several bundles' records agree on, and `bundle_input()` describes a bundle as an input of another artifact.

## Format specification

The object bundle is a folder with the following structure:

```
my_object.bundle/
    manifest.json
    data/
        ... object-specific content ...
```

- The `.bundle` suffix is **mandatory**.
- The `data/` folder contains any arbitrary representation the object chooses.
- The manifest contains all information necessary to reconstruct the object and check compatibility.

The manifest contains essential information for reconstructing the object, optional metadata, and the bundle's provenance:
```json
{
    "bundle_version": "1.1.0",
    "class_name": "MyClass",
    "class_version": "1.3.2",
    "metadata": {
        "param1": 32,
        "param2": 0.13,
        "seed": 5
    },
    "provenance": {
        "provenance_version": 1,
        "stage": "training",
        "created_at": "2026-10-07T14:03:11.482913+00:00",
        "code_commit": "d8b6d91ce2db6a0a6ccbd1e73b26920df2e1354c",
        "code_dirty": false,
        "resolved_config": {"seed": 5, "hard_budget": 3, "...": "..."},
        "seed": 5,
        "smoke_test": false,
        "method_name": "gdfs",
        "dataset_key": "cube",
        "dataset_realization_index": 0,
        "split": null,
        "inputs": [
            {
                "role": "train_dataset",
                "path": "output/production/datasets/cube/0/train.bundle",
                "class_name": "CubeDataset",
                "content_hash": "sha256:9f2c..."
            }
        ],
        "environment": {"python_version": "3.12.10", "...": "..."},
        "compute": {"device": "cpu", "...": "..."}
    },
    "content_hash": "sha256:4b1e..."
}
```

- `bundle_version`: The version of the bundle specification/protocol, as a SemVer string. `load_bundle` rejects a different major version. Version `1.1.0` added `provenance` and `content_hash`; `1.0.0` manifests, which lack both keys, still load, and `bundle_provenance()` returns `None` for them.
- `class_name`: A globally unique string that identifies the object's class; used to look up the appropriate class with `afabench.core.registry.get_class()`.
- `class_version`: The object's own version, following Semantic Versioning (SemVer). Major version differences indicate incompatibility.
- `metadata`: Optional, arbitrary information about the object. `Manifest.metadata` is an empty mapping for a manifest without it.
- `provenance`: The bundle's provenance record, below.
- `content_hash`: `sha256:<hex>` over the files under `data/` in sorted relative-path order. For each file, the relative POSIX path (UTF-8), a NUL byte, the file size as 8 big-endian bytes and the contents are fed to SHA-256; the manifest is not part of the hash. `compute_content_hash(data_path)` computes it. Consumers copy an input's hash from its manifest rather than rehashing; verifying hashes against contents is the job of the release tooling.

## Provenance

The provenance record (`CONTEXT.md`) describes how the bundle was produced. It is the frozen dataclass `afabench.core.provenance.ProvenanceRecord`, serialised with `to_json_dict()` and read with `from_json_dict()`, which rejects a `provenance_version` it does not know. `null` always means unknown, never a default. Design: `docs/adr/0002-provenance-recorded-in-artifacts.md`.

| Field | Type | Meaning |
| --- | --- | --- |
| `provenance_version` | int | Schema version, currently `1`. |
| `stage` | string | The pipeline stage that produced the bundle: `dataset_generation`, `classifier_training`, `pretraining`, `training` or `evaluation`. |
| `created_at` | string | UTC ISO-8601 time of capture. |
| `code_commit` | string or null | `git rev-parse HEAD` of the checkout holding the `afabench` package that ran, whatever the working directory; null outside a git work tree. |
| `code_dirty` | bool or null | Whether tracked files differed from `code_commit`. Untracked files are ignored. |
| `resolved_config` | object | The script's full configuration as it actually used it, after interpolation and smoke-test overrides. |
| `seed` | int | The seed actually used, never null: a null configured seed is recorded as the integer `set_seed` drew. |
| `smoke_test` | bool | Whether the bundle is a smoke-test artifact. |
| `method_name` | string or null | The pipeline's method name for training bundles; null for other stages, including pretrained models, which are shared across method names. |
| `dataset_key` | string or null | The dataset key. For a dataset bundle, the selected dataset config; for other bundles, copied from the dataset inputs' records. |
| `dataset_realization_index` | int or null | The dataset realization index, with the same sources. |
| `split` | string or null | `train`, `val` or `test` for a dataset bundle (its own split); null otherwise. |
| `inputs` | list | One entry per input bundle: `role` (`train_dataset`, `val_dataset`, `eval_dataset`, `classifier`, `pretrained_model` or `method`), `path` as given to the script, `class_name` and `content_hash` copied from the input's manifest (null for an input written before version 1.1.0). |
| `environment` | object | `python_version`, `afabench_version`, `torch_version`, `numpy_version`, `polars_version`, `lockfile_sha256` (SHA-256 of that checkout's `uv.lock`, null if absent) and `platform`. |
| `compute` | object | `device` as configured, `accelerator_name` (CUDA device name, null on CPU), `cuda_version`, `cudnn_version`, `float32_matmul_precision`, `cudnn_deterministic`, `cudnn_benchmark` and `deterministic_algorithms`. |

`capture_provenance()` collects the code, environment and compute facts itself; the caller passes the stage, resolved config, seed, smoke flag, device, inputs and identity (a `DatasetIdentity`, plus the method name and split). It raises `TypeError` for a config value that is not JSON-serialisable, naming the value. `shared_dataset_identity()` copies the dataset identity from several input records and raises `DatasetIdentityMismatchError` naming both values when they disagree; records that are null (inputs written before version 1.1.0) are unknown and cannot disagree.

Which script captures what:

| Stage | Captured by | Seed | Inputs | Identity |
| --- | --- | --- | --- | --- |
| Dataset generation | `generate_dataset.py`, `generate_image_dataset.py`, per split bundle | the realization's generation seed | none | dataset key from the selected dataset config, realization index, own split |
| Classifier training | `masked_mlp_classifier.py`, `masked_vit_classifier.py` | the seed `set_seed` returns | train and val datasets | copied from the dataset bundles' records |
| Pretraining, training | `save_result` in `afabench.fit.run` | the contract seed | datasets, classifier, pretrained model | copied from the dataset bundles' records; `method_name` from the training contract |

Tests that write fixture bundles get a record from `afabench.testing.provenance.placeholder_provenance()`.

## Registering classes for deserialization

When `load_bundle()` reads a manifest, it needs to know which Python class corresponds to the `class_name` in the manifest. This mapping is defined in `src/afabench/core/registry.py` in the `REGISTERED_CLASSES` dictionary:

```python
REGISTERED_CLASSES = {
    "MyClass": "my_module.submodule.MyClass",
    "MyDataset": "afabench.datasets.my_dataset.MyDataset",
    "RandomDummyAFAMethod": "afabench.components.methods.dummy.RandomDummyAFAMethod",
    # ... more entries ...
}
```

Each entry maps a class name to its full import path. When deserializing, the framework uses this registry to dynamically import and instantiate the correct class.

**Important:** If you create a new method class or dataset class that needs to be saved and loaded as a bundle, you must add an entry to `REGISTERED_CLASSES`.

## Implementing save and load methods

Objects that can be serialized as bundles need to implement `save()` and `load()` methods. Here's an example:

```python
from pathlib import Path
from typing import Self

class MyClass:
    def __init__(self, value: int, name: str):
        self.value = value
        self.name = name

    def save(self, path: Path) -> None:
        """Save object data to the bundle's data/ folder.

        The path parameter is the data/ folder itself.
        """
        path.mkdir(parents=True, exist_ok=True)
        import json
        data = {
            "value": self.value,
            "name": self.name,
        }
        with open(path / "data.json", "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load object from the bundle's data/ folder.

        The path parameter is the data/ folder itself.
        """
        import json
        with open(path / "data.json", "r") as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.value = data["value"]
        obj.name = data["name"]
        return obj
```

The `path` parameter passed to `save()` and `load()` is the `data/` folder itself, not the bundle root. Save and load files relative to this `path`.
