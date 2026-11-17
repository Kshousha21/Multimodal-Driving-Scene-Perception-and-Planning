"""CPU regression tests; Ray deployments and model inference are test doubles."""

import base64
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load the real application without starting Ray or requiring its installation.
ray = types.ModuleType("ray")
ray.serve = types.ModuleType("ray.serve")
ray.serve.deployment = lambda **kwargs: lambda cls: cls
handles = types.ModuleType("ray.serve.handle")
handles.DeploymentHandle = object
spec = importlib.util.spec_from_file_location("pipeline_under_test", ROOT / "pipeline_service.py")
pipeline = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"ray": ray, "ray.serve": ray.serve, "ray.serve.handle": handles}):
    spec.loader.exec_module(pipeline)

import client


def analysis(**changes):
    result = {
        "thoughts": [{"step_type": "object_identification", "frame_range": "0-1",
                      "object_code": "Object-1", "thought": "Car visible", "validation": "Observed"}],
        "verification": {"value": True, "reason": "Observed frames"},
        "isCritical": False, "confidence": 80,
        "description": "Car ahead", "goal": "Maintain distance",
    }
    result.update(changes)
    return result


def detection(box=None):
    return {"bbox": box or [0, 0, 10, 10], "cls": "car", "score": 0.9, "crop": "crop"}


def frame(index=0, dets=None):
    return {"frame_index": index, "frame": pipeline.pil_to_b64(Image.new("RGB", (24, 16))),
            "detections": [detection()] if dets is None else dets}


def planner_with_output(text):
    wrapper = pipeline.QwenPlannerWrapper.__new__(pipeline.QwenPlannerWrapper)
    wrapper.planner_agent_prompt = "Return a JSON array"
    wrapper.max_new_tokens = 50
    wrapper.temperature = 0.1
    wrapper.tok = Mock()
    wrapper.tok.apply_chat_template.return_value.to.return_value = np.array([[1, 2]])
    wrapper.tok.decode.return_value = text
    wrapper.model = Mock(device="cpu")
    wrapper.model.generate.return_value = np.array([[1, 2, 3]])
    return wrapper


class ConfigTests(unittest.TestCase):
    def test_config_and_prompts_load_outside_working_directory(self):
        cfg = pipeline.load_config(str(ROOT / "config.yaml"))
        for name in ("vision", "planner"):
            path = cfg[name]["system_prompt_path"]
            self.assertTrue(Path(path).is_absolute())
            self.assertTrue(pipeline.load_system_prompt(path).strip())


class DetectionTests(unittest.TestCase):
    def test_detector_preserves_rgb_using_pil_source(self):
        wrapper = pipeline.YoloVideoWrapper.__new__(pipeline.YoloVideoWrapper)
        image = Image.new("RGB", (16, 16), (255, 0, 0))
        wrapper._maybe_decode_video = Mock(return_value=[(0, image)])
        wrapper.model = Mock()
        wrapper.model.predict.return_value = []
        wrapper.conf, wrapper.max_det = 0.25, 200
        wrapper.infer("unused")
        self.assertIs(wrapper.model.predict.call_args.kwargs["source"], image)

    def test_invalid_frame_stride_rejected(self):
        fake = types.ModuleType("ultralytics")
        fake.YOLO = Mock()
        with patch.dict(sys.modules, {"ultralytics": fake}):
            with self.assertRaises(ValueError):
                pipeline.YoloVideoWrapper("unused", {"frame_stride": 0})

    def test_invalid_video_rejected(self):
        wrapper = pipeline.YoloVideoWrapper.__new__(pipeline.YoloVideoWrapper)
        wrapper.frame_stride = 1
        with self.assertRaises(ValueError):
            wrapper._decode_video_with_opencv(b"not a video")

    def test_real_video_decoding_and_stride(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "test.mp4")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (24, 16))
            self.assertTrue(writer.isOpened(), "OpenCV mp4v encoder unavailable")
            try:
                for _ in range(5):
                    writer.write(np.full((16, 24, 3), (0, 0, 255), dtype=np.uint8))
            finally:
                writer.release()
            wrapper = pipeline.YoloVideoWrapper.__new__(pipeline.YoloVideoWrapper)
            wrapper.frame_stride = 2
            frames = wrapper._decode_video_with_opencv(Path(path).read_bytes())
            self.assertEqual([index for index, _ in frames], [0, 2, 4])
            self.assertGreater(frames[0][1].getpixel((0, 0))[0], 240)


class TrackingTests(unittest.TestCase):
    def test_naive_assigns_unique_ids_in_same_frame(self):
        tracker = pipeline._NaiveMultiTracker()
        result = tracker.update_sequence([frame(dets=[detection(), detection([1, 1, 11, 11])])])
        ids = [item["track_id"] for item in result[0]["tracks"]]
        self.assertEqual(len(set(ids)), 2)

    def test_naive_preserves_id_across_frames(self):
        result = pipeline._NaiveMultiTracker().update_sequence([frame(), frame(1)])
        self.assertEqual(result[0]["tracks"][0]["track_id"], result[1]["tracks"][0]["track_id"])

    def test_naive_ages_tracks_through_empty_frames(self):
        result = pipeline._NaiveMultiTracker(max_age=1).update_sequence(
            [frame(), frame(1, []), frame(2)])
        self.assertNotEqual(result[0]["tracks"][0]["track_id"], result[2]["tracks"][0]["track_id"])

    def test_independent_videos_reset_naive_state(self):
        wrapper = pipeline.ByteTrackVideoWrapper.__new__(pipeline.ByteTrackVideoWrapper)
        wrapper.use_ultralytics = False
        wrapper.bt = pipeline._NaiveMultiTracker()
        wrapper.track_sequence([frame()])
        second = wrapper.track_sequence([frame(dets=[detection([12, 0, 22, 10])])])
        self.assertEqual(second[0]["tracks"][0]["track_id"], 1)

    def test_ultralytics_reset_and_real_image(self):
        wrapper = pipeline.ByteTrackVideoWrapper.__new__(pipeline.ByteTrackVideoWrapper)
        wrapper.use_ultralytics = True
        wrapper.bt = Mock()
        wrapper.bt.update.return_value = np.array([[0, 0, 10, 10, 1, 0.9, 0, 0]])
        output = wrapper.track_sequence([frame()])
        wrapper.bt.reset.assert_called_once()
        self.assertEqual(wrapper.bt.update.call_args.args[1].shape, (16, 24, 3))
        self.assertEqual(output[0]["tracks"][0]["cls"], "car")

    def test_tracker_update_errors_propagate(self):
        wrapper = pipeline.ByteTrackVideoWrapper.__new__(pipeline.ByteTrackVideoWrapper)
        wrapper.use_ultralytics = True
        wrapper.bt = Mock()
        wrapper.bt.update.side_effect = RuntimeError("tracker failed")
        with self.assertRaisesRegex(RuntimeError, "tracker failed"):
            wrapper.track_sequence([frame()])

    def test_detection_adapter_slicing_and_empty_shape(self):
        adapter = pipeline.ByteTrackVideoWrapper._DetectionsWrapper([detection(), detection()])
        self.assertEqual(len(adapter[:1]), 1)
        self.assertEqual(len(adapter[np.array([True, False])]), 1)
        self.assertEqual(pipeline.ByteTrackVideoWrapper._DetectionsWrapper([]).xyxy.shape, (0, 4))


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.wrapper = pipeline.Qwen3VLSequenceWrapper.__new__(pipeline.Qwen3VLSequenceWrapper)

    def test_empty_video_rejected(self):
        with self.assertRaises(ValueError):
            self.wrapper.infer([])

    def test_batches_cover_all_frames_and_use_final_goal(self):
        pairs = [(i, None) for i in range(51)]
        self.wrapper._infer_single_batch = Mock(side_effect=[analysis(), analysis(isCritical=True, goal="Stop")])
        output = self.wrapper._infer_batched(pairs, [frame(i) for i in range(51)])["qwen3_json"]
        self.assertTrue(output["isCritical"])
        self.assertEqual(output["goal"], "Stop")
        batches = self.wrapper._infer_single_batch.call_args_list
        self.assertEqual([len(call.args[0]) for call in batches], [50, 1])
        self.assertEqual([item["frame_index"] for item in batches[1].args[1]], [50])

    def test_failed_batch_not_verified(self):
        self.wrapper._infer_single_batch = Mock(side_effect=[analysis(), analysis(
            verification={"value": False, "reason": "Incomplete"})])
        output = self.wrapper._infer_batched([(i, None) for i in range(51)], [])["qwen3_json"]
        self.assertFalse(output["verification"]["value"])

    def test_vision_inference_failure_propagates(self):
        self.wrapper.system_prompt = "Analyze"
        self.wrapper.processor = Mock()
        self.wrapper.processor.apply_chat_template.side_effect = RuntimeError("GPU failed")
        with self.assertRaisesRegex(RuntimeError, "GPU failed"):
            self.wrapper._infer_single_batch([(0, Image.new("RGB", (8, 8)))], [])

    def test_single_batch_parses_generated_tokens_and_labels_frame_indices(self):
        class Inputs(dict):
            def to(self, device):
                return self

            @property
            def input_ids(self):
                return self["input_ids"]

        self.wrapper.system_prompt = "Analyze"
        self.wrapper.max_new_tokens = 100
        self.wrapper.processor = Mock()
        self.wrapper.processor.apply_chat_template.return_value = Inputs(input_ids=np.array([[1, 2]]))
        self.wrapper.processor.batch_decode.return_value = [json.dumps(analysis())]
        self.wrapper.model = Mock(device="cpu")
        self.wrapper.model.generate.return_value = np.array([[1, 2, 3, 4]])
        output = self.wrapper._infer_single_batch([(42, Image.new("RGB", (8, 8)))], [])
        self.assertEqual(output, analysis())
        tokens = self.wrapper.processor.batch_decode.call_args.args[0][0]
        np.testing.assert_array_equal(tokens, [3, 4])
        messages = self.wrapper.processor.apply_chat_template.call_args.args[0]
        self.assertEqual(messages[0]["content"][0], {"type": "text", "text": "Frame 42"})


class PlannerTests(unittest.TestCase):
    def test_single_step_stays_an_array(self):
        expected = [{"id": "step_1", "objective": "Stop", "success_criteria": "Stationary", "nodes_suggested": []}]
        output = planner_with_output(json.dumps(expected)).infer("Car ahead")
        self.assertEqual(output["behavior_plan"], expected)

    def test_multiple_steps_parse(self):
        expected = [{"id": "step_1"}, {"id": "step_2"}]
        self.assertEqual(planner_with_output(json.dumps(expected)).infer("Car ahead")["behavior_plan"], expected)

    def test_fenced_array_preserved(self):
        output = planner_with_output('```json\n[{"id": "step_1"}]\n```').infer("Car ahead")
        self.assertIsInstance(output["behavior_plan"], list)

    def test_invalid_planner_json_rejected(self):
        with self.assertRaises(ValueError):
            planner_with_output("not JSON").infer("Car ahead")


class IngressTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.yolo, self.bt, self.vision, self.planner = [Mock() for _ in range(4)]
        self.yolo.remote = AsyncMock(return_value={"frames": [frame()]})
        self.bt.remote = AsyncMock(return_value={"frames": [frame()]})
        self.vision.remote = AsyncMock(return_value={"qwen3_json": analysis()})
        self.planner.remote = AsyncMock(return_value={"behavior_plan": [{"id": "step_1"}]})
        self.ingress = pipeline.Ingress(self.yolo, self.bt, self.vision, self.planner)
        self.request = Mock()
        self.request.json = AsyncMock(return_value={"video": base64.b64encode(b"test video").decode()})

    async def test_stage_handoff_and_response(self):
        result = await self.ingress(self.request)
        self.assertEqual(result["yolo_detection"]["total_frames"], 1)
        self.assertEqual(result["qwen3_vision"], analysis())
        self.assertEqual(self.planner.remote.call_args.args[0]["description"], "Car ahead")
        self.assertGreaterEqual(result["processing_time"]["total_time"], 0)

    async def test_missing_video_returns_400(self):
        self.request.json.return_value = {}
        result = await self.ingress(self.request)
        self.assertEqual(result.status_code, 400)
        self.yolo.remote.assert_not_called()

    async def test_invalid_base64_returns_400(self):
        self.request.json.return_value = {"video": "not base64!!!"}
        result = await self.ingress(self.request)
        self.assertEqual(result.status_code, 400)
        self.yolo.remote.assert_not_called()

    async def test_failed_vision_does_not_run_planner(self):
        self.vision.remote.return_value = {"qwen3_json": analysis(verification={"value": False, "reason": "Failed"})}
        result = await self.ingress(self.request)
        self.assertEqual(result.status_code, 500)
        self.planner.remote.assert_not_called()

    async def test_stage_failure_returns_500(self):
        self.yolo.remote.side_effect = RuntimeError("detector failed")
        result = await self.ingress(self.request)
        self.assertEqual(result.status_code, 500)
        self.planner.remote.assert_not_called()

    async def test_malformed_json_returns_400(self):
        self.request.json.side_effect = json.JSONDecodeError("Invalid", "", 0)
        result = await self.ingress(self.request)
        self.assertEqual(result.status_code, 400)
        self.yolo.remote.assert_not_called()

    async def test_non_object_body_returns_400(self):
        self.request.json.return_value = []
        result = await self.ingress(self.request)
        self.assertEqual(result.status_code, 400)
        self.yolo.remote.assert_not_called()


class ClientTests(unittest.TestCase):
    def test_legacy_error_response_is_not_success(self):
        response = Mock()
        response.json.return_value = {"error": "detector failed", "type": "RuntimeError"}
        with patch.object(client.requests, "post", return_value=response):
            with self.assertRaises(client.requests.RequestException):
                client.send_video("http://localhost:8000", "video")

    def test_success_writes_all_stage_files_with_legacy_console(self):
        result = {
            "yolo_detection": {"frames": []}, "bytetrack_tracking": {"frames": []},
            "qwen3_vision": analysis(), "qwen_planner": {"behavior_plan": [{"id": "step_1"}]},
            "processing_time": {"total_time": 1.0},
        }
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "input.mp4"
            video.write_bytes(b"test video")
            output = Path(directory) / "output"
            response = Mock()
            response.json.return_value = result
            console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
            with patch.object(sys, "argv", ["client.py", str(video), "--output-dir", str(output)]), \
                    patch.object(sys, "stdout", console), \
                    patch.object(client.requests, "post", return_value=response):
                client.main()
            files = list(output.glob("*.json"))
            self.assertEqual(len(files), 5)
            self.assertEqual(json.loads((output / "input_qwen3_vision.json").read_text()), analysis())
            console.close()


class JsonValidationTests(unittest.TestCase):
    def test_scene_types_and_missing_fields_rejected(self):
        for changes in ({"confidence": "95"}, {"confidence": True}, {"confidence": 101},
                        {"confidence": float("nan")}, {"isCritical": "false"}, {"thoughts": []},
                        {"verification": {"value": "true", "reason": "Test"}}, {"description": ""}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                pipeline.validate_scene_analysis(analysis(**changes))

    def test_truncated_json_is_not_recovered_as_inner_object(self):
        for text in ('[{"id":"step_1"}', '```json\n[]', 'prose {"id": "step_1"}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                pipeline.parse_model_json(text)


class LauncherTests(unittest.TestCase):
    def test_launch_binds_stages_and_shutdown_only_own_application(self):
        fake_ray = types.ModuleType("ray")
        fake_ray.is_initialized = Mock(return_value=False)
        fake_ray.init = Mock()
        deployments = [Mock() for _ in range(5)]
        with patch.dict(sys.modules, {"ray": fake_ray}), \
                patch.multiple(pipeline, YoloService=deployments[0], ByteTrackService=deployments[1],
                               VisionService=deployments[2], PlannerService=deployments[3], Ingress=deployments[4]), \
                patch.object(pipeline, "serve", Mock()) as serve, \
                patch.object(pipeline.time, "sleep", side_effect=KeyboardInterrupt):
            pipeline.main(str(ROOT / "config.yaml"))
        fake_ray.init.assert_called_once()
        serve.start.assert_called_once_with(http_options={"host": "127.0.0.1", "port": 8000})
        serve.run.assert_called_once_with(deployments[4].bind.return_value,
                                          name="unexpected-situation", route_prefix="/")
        serve.delete.assert_called_once_with("unexpected-situation")
        for deployment in deployments:
            deployment.bind.assert_called_once()


if __name__ == "__main__":
    unittest.main()
