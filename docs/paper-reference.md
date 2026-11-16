# Paper reference and implementation mapping

Reference read before project selection: **A Multimodal In House Reasoning and
Planning Pipeline for Self Driving Cars in Unexpected Situations**, supplied
seven-page manuscript. No publication venue, DOI, or author list is inferred here.
The manuscript PDF is not bundled with this source-only package.

## Mapping

| Paper component | Selected code | Inspection finding |
| --- | --- | --- |
| Section II-B: YOLO11n, confidence 0.25, max 200 detections, stride 1 | `YoloVideoWrapper`, `config.yaml` | All successfully decoded frames are processed at the default stride |
| Section II-C: ByteTrack | `ByteTrackVideoWrapper` | Uses Ultralytics adapter; inherited IoU fallback is not equivalent to ByteTrack |
| Section II-D: Qwen3-VL-8B, 50-frame groups | `Qwen3VLSequenceWrapper` | Short videos use one call; longer videos use consecutive groups of 50 |
| Section II-D: merged results | `_infer_batched` | Concatenates observations/descriptions; criticality OR and maximum confidence; currently selects the first goal rather than the paper's final goal |
| Section II-E: Qwen2.5-7B planner | `QwenPlannerWrapper`, `planner_agent_prompt.py` | Emits proposed steps and node specifications, with raw-text fallback |
| Section II-F: distributed orchestration | `YoloService`, `ByteTrackService`, `VisionService`, `PlannerService`, `Ingress` | Separate actors; stages are awaited sequentially for each video |
| Introduction: executable code generation | No implementation found in selected service | Proposed future integration; no executable BT generator is supplied |
| Section II-I: quantitative evaluation | No evaluator found in selected core | No claim of reproduced detection/tracking benchmark metrics |

The code passes multiple still images to Qwen's processor rather than a native
video object. Frames are JPEG-encoded between stages, and the model processor can
resize inputs. The paper's full-resolution wording should therefore not be
interpreted as a demonstrated lossless pixel-level path.

## Results reported by the manuscript

These are manuscript claims, not measurements made during repository preparation:

| Video | Frames | Reported behavior |
| --- | --- | --- |
| Short sequence | 32 | One vision call; approximately 16 seconds end-to-end |
| Longer sequence | 827 | 17 vision batches; approximately 208 seconds vision time and roughly four minutes end-to-end |

The manuscript identifies latency, weak cross-batch temporal continuity,
daytime-focused testing, no audio, and dependence on large GPUs as limitations.
It proposes quantization, temporal caching, audio, RAG, and simulator evaluation
as future work. None of those additions is claimed by this package.

## External implementation references

- [Qwen3-VL Transformers documentation](https://huggingface.co/docs/transformers/v4.57.3/model_doc/qwen3_vl)
- [Ray Serve start API](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.start.html)
- [Ray Serve run API](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.run.html)
