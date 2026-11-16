# Known limitations and validation scope

This is a curated research snapshot. Packaging and launch portability changes
are described in `preparation.md`; the inherited model/tracking/merging algorithms
were not rewritten.

## Findings from code inspection

1. The tracker instance persists across requests and is not reset per video.
   Use a fresh service for independent evaluation videos until this is corrected.
   Its initialization can silently fall back to an IoU tracker, and update errors
   are logged and converted to empty tracks. Verify the actual tracking backend.
2. Tracking uses a fixed 30 FPS config, placeholder class IDs, nearest-IoU class
   assignment, and a hard-coded image-size tuple in the adapter. The installed
   Ultralytics API needs an integration check.
3. The detector passes an RGB NumPy array to Ultralytics. Confirm the expected
   channel order for the chosen version before measuring detector quality.
   The default 0.25 detector threshold also removes the 0.1–0.25 range before
   ByteTrack's low-confidence association stage.
4. Vision failures return fallback descriptions; inference/parsing failure does
   not consistently stop downstream planning. In particular, batch merging sets
   `verification.value` to true even if a batch reports failure. This flag must
   not be treated as proof of successful analysis or correct hazard recognition.
5. Batch merging selects `goals[0]`; the paper describes selecting the final
   batch goal. This discrepancy is retained and explicitly documented.
6. Track context summarizes ID/class/start/end frames, not detailed bounding-box
   trajectories. The same whole-video track summary is sent to each batch; prior
   batch semantic results are not carried into the next inference.
7. Planner JSON extraction tries an object regex before an array regex. Some
   multi-step arrays can parse; a single-step array may lose its outer array.
   There is no strict schema validation and invalid JSON can return raw text.
8. Empty/undecodable videos and model failures do not consistently use HTTP error
   status codes. The empty vision path has a different wrapper shape. Inspect
   output content rather than relying only on a successful HTTP response.
9. The client's `--prompt` is accepted but not consumed by ingress. GPU fractions
   are scheduling reservations, not physical-device pinning or VRAM limits.
10. Input videos and decoded frames/crops are materialized in memory, and response
    payloads repeat encoded images. Long videos can exhaust RAM/VRAM/context.
    Supplying every frame does not guarantee every event is recognized.
11. Temporary video handling uses a named open file and targets Linux behavior.
    Native Windows GPU deployment has not been validated. Debug output is written
    to `qwen3_debug.txt` in the system temporary directory.
12. The supplied descriptor/planner prompts contain strong coverage claims and
    examples; these are prompt instructions, not validated guarantees. There is
    no executable BT engine, node implementation generator, simulator, or vehicle
    control integration in the selected code.

## Validation performed during preparation

- Python syntax compilation of all packaged Python files without importing ML models.
- YAML parsing and resolution of both prompt paths relative to the config file.
- Prompt imports and non-empty `SYSTEM_PROMPT` values.
- Dependency declaration syntax and direct-import coverage.
- Mocked launch check for model/actor bindings, HTTP options, and application shutdown.
- Git ignore checks for models, datasets, videos, outputs, and credentials.
- Source provenance checks, bounded text scan, and Git whitespace checks.

Ray, PyTorch, CUDA inference, model downloads, dependency installation, tracking
quality, HTTP inference, and the manuscript's timing/quality claims were **not**
tested in this Windows preparation environment. No GPU benchmark or accuracy
claim is implied. Dependency ranges are not a tested environment lock.
