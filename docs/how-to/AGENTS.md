# How-to editing rules

Apply these rules when creating, editing, splitting or moving any guide under
`docs/how-to/`. Follow the task-based layout in
[`../README.md`](../README.md).

## Write a procedure

- Start with the task and prerequisites, then give the steps.
- Use action headings and imperative instructions. Put commands, paths and
  expected checks beside the step that needs them.
- Keep only prerequisites, decisions, actions, verification and warnings
  needed to complete the task safely.
- Link to definitions in `CONTEXT.md`, option inventories in
  `docs/reference/`, and architecture, rationale and background in
  `docs/explanation/`, rather than summarizing them in the guide.
- Keep inline clarification only when the reader needs it to choose or
  execute the next action correctly. State the required choice or warning
  directly, rather than explaining the mechanism behind it.

For example, replace “Each index seeds dataset generation, splitting and
training, so adding indices adds dataset realizations” with “Append unused
indices to `dataset_realization_indices`; retain the existing indices.”

## Review before declaring done

Review every paragraph and list item you added or changed. Identify the
specific action or safety check it enables. Remove it or move it to the
appropriate documentation type if there is none. Preserve necessary warnings
and working commands, and check links after moving detail out of a guide.
Report this procedure-only review alongside the documentation validation.
