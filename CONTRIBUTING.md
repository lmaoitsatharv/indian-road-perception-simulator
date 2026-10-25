# Contributing

Read [getting started](docs/GETTING_STARTED.md), [the code map](docs/CODE_MAP.md) and the relevant evaluation protocol before changing a module.

## Workflow

- Use descriptive topic branches such as `feature/destination-selection` or `fix/frame-alignment`.
- Keep commits focused on the resulting product or research behavior.
- Include relevant checks and state which data, model, precision and evaluation scope a result uses.
- Preserve source data, split membership, model-selection protocols and failed experiments.
- Use a fresh run directory; do not overwrite previous checkpoints or evidence.
- Keep personal notes, private study material, credentials, environments, datasets and model binaries outside source commits.

After staging changes, run `python setup/project/check_repository.py`. It inspects the staged content for common sharing mistakes. It is not a substitute for reviewing the actual diff or the affected runtime flow.

## Research results

Separate real and synthetic data, validation and fixed-model test results, and observation from control. Report class regressions and failed cases as well as aggregate gains. Reusing inspected test data for model selection changes its status; do not describe it as a fresh independent test.

Keep camera-frame identities, transforms, timestamps and input provenance explicit. A road label is not a free-space guarantee, and simulator-state control is not camera-driven control.

## Dependencies and distribution

The native simulator, TensorFlow road models, detector framework and optional MATLAB references use separate runtimes. Avoid installing every dependency into one environment. Check third-party data and software distribution terms before adding assets. No project-wide license file has been selected in this source release.
