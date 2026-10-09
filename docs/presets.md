# Removing preset effects

In the Presets panel, **Remove Preset Effects** removes recorded presets from the active photo
and keeps manual edits. The same action is in the Photo menu. Undo restores the applied look.
It does not reset the photo or disable presets for future imports.

Manual values take precedence, including changes made before, between and after preset applications.
A manually changed value remains its final value; removal does not subtract an inferred numeric
contribution. Unchanged preset masks are removed; independently added manual masks remain. A preset
mask subsequently edited by hand is retained as a manual override. Pasted or restored settings are
explicit settings overrides, rather than new preset applications.

The panel's Amount control adjusts the most recently applied preset on that photo. It keeps later
manual overrides and earlier presets, and never undoes another operation. To remove all recorded
presets, use Remove Preset Effects. Clicking a preset in the editor affects only the active photo,
even when several photos remain selected in Compare or Survey. Batch application remains available
through `preset.apply` with explicit `ids`; Auto Sync and copy/paste are explicit multi-photo actions.

Preset provenance is saved with each photo's history in catalog format 4 (with migration from either version 3 layout). Reopening, trimming history,
clearing history and virtual copies retain it. Reset returns to import defaults, including an import
preset if one was configured. Removal also supports import defaults while preserving subsequent edits.

Older histories can recover preset effects when the pre-preset snapshot is still available. If that
snapshot was never recorded or has been discarded, removal reports an actionable error and leaves
the edits unchanged. Restore a known pre-preset version or undo the preset. Older presets do not have
enough provenance to adjust Amount safely; remove and reapply them to enable it.
Incomplete old history does not block new preset applications, normal edits, history clearing or
virtual copies. A new application's Amount is available; removing all presets still requires the
missing older baseline.

Commands shared by the editor, CLI, control channel and MCP:

| Command | Behaviour |
|---|---|
| `preset.status` | Active photo's `removable`, last `preset` ID and `amount`; `error` when legacy history is incomplete |
| `preset.remove` | Remove all recorded preset effects from the active photo in one undo step |
| `preset.amount {amount: 0..200}` | Adjust the active photo's last preset, preserving manual overrides |

Regression tests exercise public engine commands and headless control input, including manual edits,
multiple presets, masks, undo/redo, import defaults, persistence, legacy history and photo switching.
