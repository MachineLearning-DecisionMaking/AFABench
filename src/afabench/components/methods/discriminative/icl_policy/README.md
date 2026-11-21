# ICL policy — official ensemble plus external acquisition masks

At each state: hide values with NaN → official TabICL views → shared
representation to BOTH heads per view → align and average scores → external
validity mask → choose ONE feature → reveal it → repeat. Ensembling does not
reveal additional features or advance the trajectory. No cluster run has
been launched for this revision.

## P01–P08

### P01 — Official input handling

**Implemented and verified.** NaN-masked numerical data enters the model interface. It uses original TabICL numerical encoding/imputation and the native EnsembleGenerator: up to 8 members, none/power normalization, Latin feature permutations, shifted classes, threshold 4.0. Tests compare predictions with the official classifier, including missing and constant columns.

**Code:** [Native ensemble preprocessing](preprocessing.py#L59).

### P02 — Same representation, external masks

**Implemented.** Within EACH view, exactly the same query-state tensor feeds the prediction decoder and policy MLP. Masks are never concatenated to the representation. We do not average embeddings before passing them to the heads.

**Code:** [Both heads within each view](model.py#L286).

### P03 — Policy scores and ensemble aggregation

**Implemented adaptation.** Each view scores feature positions. Scores are mapped back to original feature indices and averaged BEFORE the external valid-selection mask chooses one acquisition. The native filter can remove constant columns: their policy slots are placed after retained view columns and mapped to the original dropped indices. Class scores are mapped back from class shifts and averaged; default classifier temperature is 0.9. Policy-score averaging is OUR acquisition addition, not a TabICL paper claim.

**Code:** [Feature-score alignment](model.py#L287).

### P04 — Starting weights

**Implemented.** Load pretrained TabICL strictly; initialize policy weights freshly. Each dataset realization starts independently. Joint mode fine-tunes all originally trainable pretrained components; fixed RoPE constants remain fixed.

**Code:** [Pretrained load](model.py#L106).

### P05 — Valid acquisitions

**Implemented.** Externally exclude acquired/unavailable features; handle exhausted queries without invalid softmax. ONE original feature is selected per query/state and revealed consistently across every view. Unit-cost individual features only.

**Code:** [External validity mask](train.py#L90).

### P06 — Learning

**Implemented and tested.** Straight-through Gumbel-softmax on combined policy scores; predict after the exact hard raw reveal in all views; average aligned class scores for CE. Backward uses a prepared-endpoint reveal surrogate, not derivatives through sklearn. Random selections separately advance training states; gradients of the same normalized loss are accumulated per acquisition step, freeing each graph before the next step. Adam still updates once per whole batch. This bounds trajectory memory without reducing batch size or acquisition count. Fixed labeled context differs from L2M autoregressive context.

**Code:** [Same reveal across views](model.py#L192).

### P07 — Updates and restart

**Implemented.** Policy 1e-4; pretrained trainable components 1e-5 in joint mode. Frozen mode is a baseline. Save complete latest/best model, both Adam groups and RNG. Old single-view checkpoints are incompatible; use new ensemble output roots.

**Code:** [Optimizer groups](train.py#L129).

### P08 — Minerva readiness

**GPU pilot still required.** Local tests cover official one/eight-view prediction parity, class/feature alignment, variable native view counts, hard-reveal parity, selection gradients and joint updates. GPU memory/throughput and full repository QA remain unverified. Ensembling increases repeated forward/backward cost; do not assume the old pilot timings apply.

**Code:** [Run commands](../../../../../../experiments/icl_policy/README.md).

## Checks

```sh
uv run pytest test/src/afa_discriminative/test_icl_policy.py test/scripts/test_icl_policy_run.py -q -n 0
```

Files: `model.py` connects the views/heads; `policy.py` is the embedding-only
MLP; `preprocessing.py` adapts native ensemble generation; `train.py` owns
external masks, loss and optimizers; `tasks.py` samples L2M-style BNN tasks.
Original source lives under `upstream/`; no new upstream source patch was
needed for the ensemble. Production settings are in
`experiments/icl_policy/config.json`.

Class-slot permutations reserve all declared task classes, including classes
absent from a generated context. Dummy slot IDs only configure the native
generator's class count; actual context labels are transformed into every
view. Policy outputs retain original feature count/order across acquisitions.
