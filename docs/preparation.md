# Repository preparation

## Source selection

The input contained three project variants:

| Original folder | Role | Selection |
| --- | --- | --- |
| `unexpected_situation` | Integrated Qwen3-VL pipeline with consecutive 50-frame batching and expanded prompts | Selected core |
| `test_project` | Earlier integrated version, alternate fixed service, and model experiment | Excluded as duplicate/earlier experiments |
| `Unexpected Situation` | Earlier standalone detector/tracker code, planner notebook, prompt archives, dataset, and third-party ByteTrack tree | Excluded from the core package |

The source originals were read only. Four essential Python files were copied.
Both prompts remain byte-identical to their selected originals; the client and
service now include the fixes described in `verification.md`.
`source-manifest.json` records historical hashes of the selected originals, not
checksums of the current working files.

## Changes in this copy

- Replaced mixed MiniCPM/Qwen legacy documentation with a project-specific README
  and an explicit paper-to-code mapping.
- Replaced machine-specific prompt paths with config-relative paths resolved
  before Ray actor creation.
- Replaced the launcher with public `serve.start(http_options=...)` and
  `serve.run(...)` calls. It no longer shuts down all existing Serve applications
  before deployment; Ctrl+C deletes only this named application.
- Replaced hard-coded `/tmp` debug file paths with the system temporary directory.
- Defaulted HTTP to localhost and attention to SDPA. FlashAttention is optional.
- Corrected tracking option names in the example config to names the wrapper reads.
- Removed the unused `max_images` option from the example config; batching is
  actually fixed at 50 in the code. Added PyYAML and requests to dependencies,
  removed unused torchaudio/qwen-vl-utils, and separated optional FlashAttention.
- Added `.gitignore` and `.gitattributes`; initialized a new local Git repository
  on `main`. No commit, remote, push, or GitHub repository was created.

## Excluded files

- Model weights, input videos, the full dataset, generated frames/crops, and
  stage-output JSON files.
- ZIP/tar archives, caches, compiled binaries, notebook outputs, and duplicate services.
- Legacy upload script containing a specific remote server address.
- Old setup/cleanup scripts, stale migration/checklist/fix notes, and generated paper text.
- The old `serve_config.yaml`, which referenced a module-level `ingress` symbol
  that the original module did not export. Use the documented Python launcher.
- The old `test_batch.py`, which simulated a different 100-frame batch size and
  did not exercise the service's implementation.

The reference manuscript was read and summarized but is not redistributed here.
No additional project license or publication identifiers were invented.

## Push to GitHub when ready

Create an empty repository in your GitHub account, then run from this folder:

```bash
git status --short
git add .
git diff --cached --stat
git commit -m "Add offline multimodal driving analysis and planning pipeline"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace both URL placeholders. Review `known-limitations.md` before presenting
the repository as a working reproduction. When copying the ZIP contents, run
`git init -b main` first because the ZIP excludes local Git metadata.
