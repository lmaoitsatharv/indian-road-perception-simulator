# Security and responsible use

This repository is a research prototype for road perception and traffic simulation. Its published results describe specific data and synthetic scenarios; they do not establish suitability for controlling a physical vehicle or general road safety. The current simulator driver uses rules and simulator states. Camera-based vehicle control and reinforcement learning are not implemented.

See [Getting started](docs/GETTING_STARTED.md), [Results](docs/RESULTS.md) and the [documentation index](docs/INDEX.md) for operational requirements and measured limitations.

## Repository contents

Source code, clean notebooks, dependency specifications, model metadata, original small scenery assets and curated aggregate/synthetic evidence belong in version control. Credentials, `.env` values, access tokens, private keys, virtual environments, engine binaries, datasets, downloaded footage, model weights and full runtime output must remain outside commits. Preserve dataset and media licensing/access requirements separately from the source repository.

`.gitignore` helps prevent accidental additions but does not remove sensitive material already tracked. Before publishing changes, run the staged-content check from the repository root:

```text
python setup/project/check_repository.py
```

It examines staged bytes for common credential patterns, excluded paths, unexpected binary/large files, notebook output and documented scenery-texture hashes. It is a publication guard, not a complete vulnerability assessment. Review the staged diff as well. If a real credential is exposed, revoke or rotate it immediately; deleting the current file alone does not remove the exposure.

## Local execution and network boundaries

- Run the simulator and local prediction services on a trusted machine and network. Keep CARLA RPC and preview/service endpoints bound to loopback or a deliberately restricted local network; do not expose them directly to the public internet.
- Use the existing owned-session launchers and cleanup paths. Do not replace scoped process ownership with broad termination of every Python, MATLAB or CARLA process.
- Keep frame/session identity, packet-size limits, bounded buffering and stale-result handling enabled in the live observation path. A result computed successfully can still be too old to use.
- Review scripts before execution and install dependencies/engine distributions from their official sources. Use isolated environments for the segmentation, detector and simulator runtimes where their package constraints differ.
- Treat downloaded archives and annotations as untrusted input. Preserve path, size, coordinate, class and provenance validation during extraction and conversion.

## Model and data verification

Load only trusted checkpoints and verify their SHA-256 values against [the model metadata](models/README.md). Some model deserialization formats can execute code, so a familiar extension is not a security boundary. A recorded local hash establishes reproducibility/content identity; it is not a publisher signature or a substitute for a trusted source.

Retain the explicit separation between locked real-road inference, simulator-only observation and the object detector. A manifest for observation does not authorize actuator commands. Keep original data immutable, preserve split identities and record exclusions. Unknown labels, missing annotations, uncertain synthetic pixels and stale observations must not silently become negative examples or free space.

Do not publish third-party photographs, footage or annotations without checking their distribution rights and any applicable consent. Screenshots and logs may also reveal personal paths, access URLs or credentials; review their contents before attaching them to a public issue.

## Contributions and reporting

Use a feature branch and review changes before merging. Include reproducible steps, environment versions, relevant configuration and bounded sanitized logs when reporting a defect. Distinguish a source-code check, an offline fixture, a saved-result audit and an executed native scenario when describing validation.

Report suspected credential exposure or exploitable vulnerabilities to the maintainers through an available private reporting channel. Do not disclose secret values or operational exploit details in a public issue before coordination. No dedicated email address or guaranteed response time is specified here.

Repository checks do not constitute a completed formal security audit. Finite collision-free simulator runs, high model scores and successful startup checks do not establish safe real-world vehicle operation.
