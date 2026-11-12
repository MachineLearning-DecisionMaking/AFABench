# AFABench
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.14734-b31b1b.svg)](https://arxiv.org/abs/2508.14734)
[![KDD 2026](https://img.shields.io/badge/KDD-2026-ff69b4.svg)](https://www.kdd.org/kdd2026/)

**A benchmark for Active Feature Acquisition (AFA) methods**

Compare state-of-the-art algorithms for sequential feature selection in
scenarios where acquiring features is costly. Includes implementations of
multiple AFA methods, standardized datasets, and automated evaluation pipelines.

## Start here

- [Set up AFABench](docs/how-to/setup.md).
- [Run the pipeline](docs/how-to/run_pipeline.md).
- [Understand and customize it](docs/explanation/pipeline.md).
- [Run on SLURM](docs/how-to/run_on_slurm.md).
- [Download or extend published results](docs/how-to/manage_releases.md).

## Features

- Accessible configuration using
  [hydra](https://hydra.cc/)
- Reproducible pipeline using [snakemake](https://snakemake.readthedocs.io/en/stable/).
- Modular design: rerun specific parts of the pipeline as needed.
- Extensible framework: add custom datasets and AFA methods.

## Limitations
- Supports only classification tasks; regression tasks are not yet implemented.

## What is Active Feature Acquisition?
**Active Feature Acquisition (AFA)** addresses scenarios where,

- **Features are expensive** to obtain (medical tests, surveys, sensors),
- **Real-time decisions** must be made with partial information,
- **Budget constraints** limit which features you can acquire.

**Example**: Medical diagnosis where each test costs money and time. AFA methods
intelligently decide which tests to order next based on previous results, aiming
for accurate diagnosis with minimal cost. See the following survey for details: [AFA Survey](https://arxiv.org/abs/2502.11067).

## Implemented Methods
|    Method     |                                                                            Paper                                                                             |             Strategy             |  Greedy?   |
| :-----------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------: | :--------: |
|  **EDDI**  |                                                       [link](https://proceedings.mlr.press/v97/ma19c)                                                        |   Generative estimation of CMI   |   Greedy   |
|  **GDFS**  |                                                     [link](https://proceedings.mlr.press/v202/covert23a)                                                     | Discriminative estimation of CMI |   Greedy   |
|  **DIME**  |                                                           [link](https://arxiv.org/pdf/2306.03301)                                                           | Discriminative estimation of CMI |   Greedy   |
|  **L2M**  |                                                           [link](https://arxiv.org/abs/2510.12624)                                                           | Discriminative estimation of CMI, in-context |   Greedy   |
| **JAFA** |                          [link](https://papers.nips.cc/paper_files/paper/2018/hash/e5841df2166dd424a57127423d276bbe-Abstract.html)                           |          Model-free RL           | Non-greedy |
|  **OL**  |                                                           [link](https://arxiv.org/pdf/1901.00243)                                                           |          Model-free RL           | Non-greedy |
| **ODIN** | [link](https://www.microsoft.com/en-us/research/publication/odin-optimal-discovery-of-high-value-information-using-model-based-deep-reinforcement-learning/) |          Model-free RL           | Non-greedy |
| **ODIN** | [link](https://www.microsoft.com/en-us/research/publication/odin-optimal-discovery-of-high-value-information-using-model-based-deep-reinforcement-learning/) |          Model-based RL          | Non-greedy |
|   **AACO**    |                                                 [link](https://proceedings.mlr.press/v235/valancius24a.html)                                                 |           Oracle-based           | Non-greedy |
|   **PT**    |                                              [link](https://link.springer.com/article/10.1023/A:1010933404324)                                               |    Global feature importance     |    N/A     |
|   **CAE**   |                                                   [link](https://proceedings.mlr.press/v97/balin19a.html)                                                    |    Global feature importance     |    N/A     |

## Datasets
Each dataset's source, license, instances and features are described on its
page in [`docs/reference/datasets/`](docs/reference/datasets/README.md).

| Dataset | Type | Modality | Train Size | Val Size | Test Size | \# Features | \# Groups | \# Classes |
| :-----: | :--: | :------: | :--------: | :------: | :-------: | :---------: | :-------: | :--------: |
| [CUBE](docs/reference/datasets/cube.md) | Synthetic | Tabular | 600 | 200 | 200 | 20 | 20 | 8 |
| [CUBE-NM](docs/reference/datasets/cube_nm.md) | Synthetic | Tabular | 600 | 200 | 200 | 55 | 51 | 8 |
| [MNIST](docs/reference/datasets/mnist.md) | Real World | Image (tabularized) | 36,000 | 12,000 | 12,000 | 784 | 784 | 10 |
| [Fashion-MNIST](docs/reference/datasets/fashion_mnist.md) | Real World | Image (tabularized) | 36,000 | 12,000 | 12,000 | 784 | 784 | 10 |
| [Diabetes](docs/reference/datasets/diabetes.md) | Real World | Tabular | 55,237 | 18,412 | 18,413 | 45 | 45 | 3 |
| [PhysioNet Challenge 2012](docs/reference/datasets/physionet_challenge_2012.md) | Real World | Tabular | 7,200 | 2,400 | 2,400 | 41 | 41 | 2 |
| [MiniBooNE](docs/reference/datasets/miniboone.md) | Real World | Tabular | 77,755 | 25,918 | 25,919 | 50 | 50 | 2 |
| [MiniBooNE subset](docs/reference/datasets/miniboone_subset.md) | Real World | Tabular | 77,755 | 25,918 | 25,919 | 10 | 10 | 2 |
| [ACTG 175](docs/reference/datasets/actg.md) | Real World | Tabular | 1,283 | 427 | 429 | 23 | 23 | 2 |
| [Chronic Kidney Disease](docs/reference/datasets/ckd.md) | Real World | Tabular | 240 | 80 | 80 | 24 | 24 | 2 |
| [Bank Marketing](docs/reference/datasets/bank_marketing.md) | Real World | Tabular | 27,126 | 9,042 | 9,043 | 16 | 16 | 2 |
| [Imagenette](docs/reference/datasets/imagenette.md) | Real World | Image | 5,681 | 3,788 | 3,925 | 150,528 | 196 | 10 |

## Project structure
- `src/afabench/`: Importable AFABench library, included in the Python wheel.
- `docs`: Documentation.
- `conf`: All the **script** configuration files. Each configuration file
  corresponds to a class in `config_classes.py`.
- `data`: One folder per dataset key, with its committed source and license (ADR 0014).
- `workflow/`: Repository-only benchmark pipeline and its launcher, `workflow/run.py`.
- `output`: Outputs from the Snakemake pipeline (gitignored).
- `logs`: W&B, Lightning and Hydra logs (gitignored).
- `scripts/`: Folder of scripts, many of which are called from the snakemake pipeline.
- `test`: Tests.
  - `src`: Tests related to library code in `afabench`.
  - `scripts`: Tests related to specific scripts in `scripts`.

## Documentation

See the [documentation index](docs/README.md) for guides and references,
including [adding a dataset](docs/how-to/add_dataset.md) or
[adding a method](docs/how-to/add_method.md).

## Development

We encourage researchers to fork this repository and implement their own methods. Take a look at the [documentation](#documentation) to get started.

To follow repo conventions, run
```shell
uv run just qa
```
which will

- format code with `ruff format`
- do linting and formatting with `ruff check --fix`
- type checking with `basedpyright --warnings`
- run tests with `pytest`, adding the Snakemake workflow tests when your
  changes touch the workflow (`uv run just qa-full` always runs them)

## Citation
If you use this benchmark in your research, please cite,
```bibtex
@inproceedings{Schütz2026,
    author = {Sch{\"u}tz, Valter and Wu, Han and Rezvan, Reza and Aronsson,
              Linus and Haghir Chehreghani, Morteza},
    title = {AFABench: A Generic Framework for Benchmarking Active Feature
             Acquisition},
    year = {2026},
    booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge
                 Discovery and Data Mining},
    url = {https://doi.org/10.1145/3770855.3817493},
}
```
