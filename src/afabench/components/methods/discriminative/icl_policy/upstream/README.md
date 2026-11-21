# Vendored TabICL model code

Copied from the installed TabICL 2.2.0 package, pinned to the authors’
repository https://github.com/soda-inria/tabicl at commit
`c91f00df184a5097e584cb376f7e699e9f3444c4` (installed direct_url.json).
The BSD 3-Clause license is retained in LICENSE.

`source_manifest.json` records each original path and SHA-256 before patches.
All `_model/*.py` files are copied together with `_torch_devices.py`.

The only source patch is marked `PATCH(afabench)` in `inference.py`: change
`.._torch_devices` to `._torch_devices` because the helper is copied into
this folder. No numerical/model behavior is changed.

As with l2m/upstream, these sources are excluded from repository
formatting, lint and type checks to preserve upstream comparison. The
adapter and policy beside this folder follow repository standards.

Model map:
- `tabicl.py`: complete model construction and original forward paths.
- `embedding.py`: column embedding.
- `interaction.py`: row interaction.
- `learning.py`: label encoder, TF_icl and original classifier decoder.
- `encoders.py`, `layers.py`, `attention.py`, `ssmax.py`: transformer blocks
  and attention internals.
- Remaining files: supporting inference, cache, device and distribution code.

The sklearn classifier/ensemble interface is not copied. Its numeric
`preprocessing.py` and `sklearn_utils.py` are now copied unchanged, so our
model interface can call the actual authors’ preprocessing components. PyTorch,
NumPy and other existing runtime dependencies are still required.
The policy experiment does not import the installed tabicl package;
reference tests use it only to check equivalence.

## Explicit change register (verified 19 November 2026)

| File | Marked change | Reason | Model behavior |
|---|---|---|---|
| `inference.py` | `# PATCH(afabench)` above the device-helper import | Change `.._torch_devices` to `._torch_devices` for this folder layout | No numerical change |

Every other Python file was checked against the original hashes in
`source_manifest.json` and is unchanged. Reversing the one documented import
patch reproduces its original hash too. The manifest records the originals,
not hashes regenerated from our modified copies.

**Our method changes are outside upstream:** `../policy.py` supplies the
embedding-only policy; `../model.py` connects both heads and enables joint
or frozen training; `../train.py` supplies external selection masks and two
learning-rate groups. The experiment runner saves and restores the complete
adapted model for each dataset realization. Jointly updating pretrained
weights does not require editing the upstream layer definitions.

Any future source edit here must carry a nearby `# PATCH(afabench):` comment
explaining the reason and be added to this table, including whether it
changes numerical behavior. Do not silently reformat these source copies.

### Added original numeric preprocessing sources

`preprocessing.py` and `sklearn_utils.py` were copied unchanged from
`src/tabicl/_sklearn/` at the pinned installed revision. Their original
hashes are recorded in source_manifest.json. No source patches were needed.
The only patched upstream Python source remains the device-helper import
in inference.py. Numeric selection/reveal adaptation lives outside upstream,
in ../preprocessing.py and ../model.py.
