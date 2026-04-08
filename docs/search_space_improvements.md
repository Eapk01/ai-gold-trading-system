# Search Space Improvement Notes

This document lists only the suggested improvements for making the research search space easier to understand, extend, and edit.

The goal is simple:

- a developer should know where to add or change search options immediately
- the GUI should discover available options from one clear source of truth
- adding a target, preset, trainer, or feature set should not require hunting through multiple files

## 1. Make Search Options Discoverable From Registries

### Suggested improvement

Replace scattered hard-coded lists with explicit registries for:

- Stage 5 target specs
- Stage 5 preset definitions
- trainer definitions
- named feature sets

### Why

Right now, some options are defined in one backend file and then re-listed in GUI code or helper code. That creates a "change it in two places" problem.

### Desired shape

Each option type should have one canonical source of truth with:

- internal id
- display name
- description
- compatibility rules if needed
- actual payload or builder

### Result

The GUI can read from those registries directly instead of maintaining its own lists.

## 2. Move Stage 5 Target Options Out Of Service Wiring

### Suggested improvement

Move Stage 5 target option creation out of `ResearchWorkflowService._build_stage5_search_target_specs()` and into a dedicated target registry module.

### Why

The current service method mixes workflow orchestration with "what targets exist." That makes extending targets harder than it should be.

### Desired shape

Have a single place that returns all available search targets, with enough metadata for both:

- backend execution
- GUI display

### Result

Adding a new target should be a one-file change, and the GUI should automatically see it.

## 3. Make Presets Fully Data-Driven In The UI

### Suggested improvement

Use the preset registry as the only source for:

- available preset names
- display names
- descriptions
- trainer-specific parameter payloads

### Why

Preset definitions already live in one backend module, but the UI still has explicit preset lists in a few places. That means adding a new preset in code does not fully show up in the app automatically.

### Desired shape

The UI should ask the backend for the full preset catalog and render whatever comes back.

### Result

Adding a preset should require only:

- adding the preset definition once
- optionally selecting it in config or GUI

## 4. Separate Feature Classification From Feature Set Definitions

### Suggested improvement

Split:

- feature classification rules
- named feature set definitions

into clearer layers.

### Why

Right now, feature groups and named sets are tightly coupled. That makes the code harder to scan and makes future custom feature sets more awkward.

### Desired shape

Keep low-level classification logic in one place, and keep named feature set construction in another place that is easier to extend.

### Result

A developer can add a new named feature set without needing to understand the full grouping pipeline.

## 5. Add A Search Catalog API For The GUI

### Suggested improvement

Expose one service method that returns the full editable search catalog, for example:

- available targets
- available feature sets
- available trainers
- available presets
- current defaults
- compatibility warnings

### Why

The GUI currently pulls pieces of this information from multiple places. A single service payload would make the UI much simpler and reduce duplicate logic.

### Desired shape

The GUI should render from one response object rather than rebuilding backend knowledge on the page.

### Result

The page becomes smaller, clearer, and safer to change.

## 6. Support One-Off Search Overrides Separately From Saved Defaults

### Suggested improvement

Separate:

- "run this search now with these selections"
- "save these selections as config defaults"

into two different actions.

### Why

Editing config should not be required just to try a different search space once.

### Desired shape

The backend should accept optional runtime overrides for Stage 5 search settings, while config remains the persistent default layer.

### Result

The GUI can become editable without forcing config writes on every experiment.

## 7. Standardize Option Metadata Across Targets, Presets, Trainers, And Feature Sets

### Suggested improvement

Use a consistent metadata shape for anything shown in the GUI.

### Include

- `id`
- `display_name`
- `description`
- `is_default` or selected/default status when relevant
- compatibility metadata when relevant

### Why

Right now, each option type exposes slightly different information. A shared shape would make rendering much easier.

### Result

GUI components become reusable instead of custom for each search option type.

## 8. Make The GUI Purely Reflective Where Possible

### Suggested improvement

Avoid hard-coding visible search choices in Streamlit pages.

### Why

The GUI should present what the backend supports, not its own duplicated version of that support.

### Desired rule

If a new preset, trainer, target, or feature set is added to the backend registry, the GUI should show it automatically unless there is a deliberate product reason not to.

### Result

Most future search-space expansion becomes additive rather than refactor-heavy.

## 9. Keep Workflow Services Focused On Orchestration

### Suggested improvement

Keep workflow services responsible for:

- building requests
- validating selections
- running the search

and move catalog-building responsibilities into dedicated helper modules.

### Why

Service classes get harder to reason about when they also own the definition of all user-facing options.

### Result

The codebase becomes easier to navigate:

- registries define what exists
- services decide how to run it
- GUI decides how to display it

## 10. Add Small Developer-Facing Documentation Near The Extension Points

### Suggested improvement

Wherever the final extension points live, add short comments explaining:

- where to add a new preset
- where to add a new target
- where to add a new feature set
- what the GUI auto-discovers
- what still requires explicit wiring

### Why

Even a good structure becomes frustrating if the entry points are not obvious.

### Result

A future change should feel like:

1. open the registry file
2. add the new option
3. run the app

not like a codebase treasure hunt.

## Recommended Direction

If these improvements are implemented, the system should follow this rule:

> Definitions live in registries, defaults live in config, runtime overrides live in request objects, and the GUI reads from service-provided catalogs.

That gives a clear mental model and keeps changes localized.
