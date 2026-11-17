# Known limitations and validation scope

This is an offline research prototype. CPU regression tests cover application
logic and video decoding; they do not validate model quality or GPU deployment.
See [verification](verification.md) for the latest results.

## Corrected during the error review

- Reset tracking state for each video and prevent the naive tracker assigning
  the same ID to multiple detections in a frame.
- Supply a real BGR frame to ByteTrack, fix adapter slicing/empty-array shapes,
  report fallback initialization, and propagate tracking update failures.
- Pass RGB PIL images directly to YOLO instead of RGB NumPy arrays.
- Close temporary video files before OpenCV reopens them on Windows, always
  release the decoder, and reject videos with no decodable frames.
- Preserve JSON plan arrays, reject malformed model responses, validate the
  scene fields used downstream, and stop planning on unverified analysis.
- Preserve failed batch verification and select the final batch goal. Each
  batch receives only its own track summary and explicit image frame indices.
- Return HTTP error responses for invalid requests and pipeline failures. The
  client also detects legacy error objects returned with HTTP 200.
- Remove emoji console status messages and unconditional debug-file writes.

## Remaining limitations

1. The actual Ultralytics, Ray, Transformers, PyTorch, CUDA, and model-weight
   combination has not been installed or exercised here. Tests use Ray/model
   doubles. Dependency ranges are not a validated environment lock.
2. Tracking uses configured 30 FPS rather than source-video FPS, placeholder
   class IDs, and nearest-IoU class/crop assignment. The naive IoU fallback is
   not equivalent to ByteTrack. Non-default frame strides change tracking time
   semantics. Tracking accuracy still requires evaluation with real footage.
3. The detector's default 0.25 threshold removes the 0.1-0.25 confidence range
   before ByteTrack's low-confidence association stage.
4. Track context summarizes ID, class, and start/end frames rather than detailed
   bounding-box trajectories. Prior batch semantic analysis is not carried into
   the next model call. A final-batch goal can differ from earlier hazards;
   criticality and descriptions retain information from all batches.
5. Scene validation checks the top-level fields consumed by the pipeline, not
   every nested observation. Planner validation requires a non-empty array of
   objects but does not validate every plan-step or node field. Valid JSON is
   not proof of a sensible or feasible plan. The verification flag is model
   output, not independent verification of scene recognition.
6. The client's `--prompt` is accepted but is not consumed by ingress. GPU
   fractions are scheduling reservations, not physical-device pinning or VRAM
   limits. The default deployments reserve 2 GPUs and 22 CPUs in total.
7. Videos, decoded frames, and crops are held in memory; response payloads repeat
   encoded images. Long videos can exhaust RAM, VRAM, or model context. All-frame
   input coverage does not guarantee recognition of every event.
8. Native Windows GPU deployment has not been validated. The local test suite
   verifies video decoding and console compatibility on Windows only.
9. The supplied prompts contain strong coverage claims and inconsistent planning
   examples. These are instructions, not measured guarantees. There is no
   executable behavior-tree engine, node generator, simulator, or vehicle-control
   integration in this repository.

The manuscript's latency and quality claims have not been reproduced. Use a
provisioned Linux GPU host for model integration tests before evaluating outputs.
