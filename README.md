# Unexpected Situations: Multimodal Driving Analysis and Planning

An offline research pipeline that turns traffic video into object detections,
tracked objects, structured scene analysis, and proposed behavior-tree plan steps.

Prepared from the project accompanying **A Multimodal In House Reasoning and
Planning Pipeline for Self Driving Cars in Unexpected Situations**.
The code processes decoded frames with stride 1 and submits consecutive groups
of up to 50 images to the vision model. This is full-frame input coverage;
it does not establish that the model recognizes every event.

## Pipeline

| Stage | Implementation | Output |
| --- | --- | --- |
| Detection | Ultralytics YOLO11n | Frame images, boxes, classes, scores, crops |
| Tracking | Ultralytics ByteTrack, with an inherited simple IoU fallback | Track IDs and boxes per frame |
| Scene analysis | Qwen3-VL-8B-Instruct | Structured observations, criticality, description, goal |
| Planning | Qwen2.5-7B-Instruct | Proposed plan steps and custom node specifications |

The planner produces JSON specifications. An executable behavior-tree generator,
vehicle controller, and simulator integration are not included.

## Files

| File | Purpose |
| --- | --- |
| `pipeline_service.py` | Model wrappers, Ray deployments, and HTTP ingress |
| `client.py` | Submit a local video and save each stage's response |
| `system_prompt.py` | Scene descriptor prompt |
| `planner_agent_prompt.py` | Behavior planning prompt |
| `config.yaml` | Models, prompt paths, detection/tracking options, HTTP settings |
| `requirements.txt` | Core dependency ranges |
| `requirements-optional.txt` | Optional FlashAttention dependency |
| `docs/paper-reference.md` | Paper-to-code mapping and reported results |
| `docs/known-limitations.md` | Findings from source inspection and validation scope |
| `docs/preparation.md` | Selection decisions, exact changes, and GitHub commands |
| `docs/source-manifest.json` | Checksums for the selected original files |

## Installation

Use a Linux GPU environment; Python 3.11 is a suggested starting point, not a
validated lock. The paper reports Ubuntu and two A100 GPUs (80 GB each), with
64 GB or more system RAM. This preparation did not reproduce that environment.
The deployments reserve 2 GPUs in total and 22 CPUs, plus ingress/Ray overhead.
Resource and memory options are currently declared in `pipeline_service.py`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install CUDA-enabled PyTorch and torchvision using the command appropriate for
your driver from [PyTorch's installer](https://pytorch.org/get-started/locally/),
then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Weights are excluded from Git. Supply `yolo11n.pt` in the repository directory
or allow Ultralytics to download it. The Qwen wrappers use Hugging Face model
identifiers and download/cache weights on first load. Allow sufficient disk
space for both models. Dependency ranges still require GPU integration testing.

The default config uses SDPA. For optional FlashAttention, install
`requirements-optional.txt` after PyTorch and select `flash_attention_2` in the config.

## Run

From the repository directory on a suitably provisioned GPU host:

```bash
python pipeline_service.py config.yaml
```

In another activated terminal, submit a video you have permission to use:

```bash
python client.py /path/to/traffic-video.mp4 --server http://127.0.0.1:8000 --timeout 1800 --output-dir output
```

An existing Ray Serve instance retains its HTTP options; use a fresh dedicated
Ray environment if its host/port differs. The launcher uses the public
[`serve.start`](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.start.html)
and [`serve.run`](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.run.html) APIs.
The service binds to localhost by default. To reach a remote GPU machine, forward
its port through your own SSH connection and keep the client pointed at localhost.

The client writes detection, tracking, vision, planner, and combined JSON files.
Detection/tracking JSON contains encoded frames/crops and can be large; these
outputs are ignored by Git. Check the response for errors and read the known
limitations before interpreting any result. The client's inherited `--prompt`
argument is currently ignored by the server.

## Reference and status

Read [the paper mapping](docs/paper-reference.md) and
[known limitations](docs/known-limitations.md). Original prompts and research
logic are retained; packaging does not imply full runtime validation or measured
performance reproduction. The paper's reported latency makes this an offline
analysis prototype, unsuitable for direct real-time vehicle control.

No project license was selected during preparation. Choose the appropriate
project license before distributing it under a specific license, and retain
applicable dependency/model license terms. No third-party source checkout,
dataset, or model weights are bundled.

For publication steps, see [preparation and GitHub commands](docs/preparation.md).
