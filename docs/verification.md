# Project verification

Checked on 2026-11-16 in Windows with Python 3.14.0.

## Latest automated checks

Run:

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python -m compileall -q client.py pipeline_service.py system_prompt.py planner_agent_prompt.py tests
git diff --check
```

The regression suite contains 33 tests covering:

- YAML loading, config-relative paths, and both prompt imports.
- A real generated five-frame MP4 decoded through OpenCV, frame stride, RGB
  preservation, and rejection of undecodable input.
- Detector PIL input and rejection of zero frame stride.
- Naive tracking association, unique IDs, aging, and reset between videos.
- ByteTrack adapter slicing, empty-array shape, reset, real frame input, and
  propagation of update errors, using a tracker test double.
- Complete 50+1 batch coverage, batch-specific track summaries, final-goal
  selection, and preservation of failed verification.
- Vision inference failure propagation, generated-token trimming, explicit
  frame labels, and model-response parsing with processor/model test doubles.
- Single-step and multi-step planner arrays, fenced JSON, truncated/invalid
  responses, and malformed scene field types.
- Ingress stage handoff, malformed/missing input, real HTTP response status
  objects, and stopping the planner after failed scene verification.
- Client handling of legacy error objects and writing all five JSON outputs
  with a CP1252 console.
- Mocked launcher bindings, HTTP options, and shutdown scoped to this app.

The initial 26-test regression run reported 14 failures and 8 errors before
fixes. Some errors came from unused Ultralytics imports and Windows console
encoding. The final suite passes all 33 tests.

Syntax compilation, client CLI help, installed-package consistency (`pip check`),
and Git whitespace checks also pass. The installed-package check only covers
packages already present; it does not mean the full requirements are installed.

## Changes confirmed by these tests

See [known limitations](known-limitations.md) for the corrected defects and
remaining limitations. YOLO now uses PIL input, as supported by the
[Ultralytics prediction input contract](https://docs.ultralytics.com/modes/predict/).
The tracker image/reset adapter follows the
[ByteTrack reference](https://docs.ultralytics.com/reference/trackers/byte_tracker/).
The raw ingress returns Starlette response objects for custom status codes,
following the [Ray HTTP guide](https://docs.ray.io/en/latest/serve/http-guide.html).

## Not exercised

Ray, PyTorch, Transformers, and Ultralytics are not installed in this environment.
No weights were downloaded, and no real model inference, actual Ray HTTP service,
CUDA execution, tracking/detection benchmark, or manuscript performance
reproduction was run. These require the Linux GPU setup documented in README.md.
The tests do not prove compatibility of the declared dependency ranges.

## Historical preparation checks

The repository preparation previously checked dependency declaration syntax,
source provenance, exclusion rules for weights/data/secrets, and a bounded text
scan. Those historical checks did not execute any ML models. Source-manifest
hashes describe the original selected sources, not the now-modified service and
client.
