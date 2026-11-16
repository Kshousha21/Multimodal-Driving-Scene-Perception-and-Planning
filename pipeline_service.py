"""Offline detection, tracking, scene analysis, and behavior-plan generation.

Run from the repository: python pipeline_service.py config.yaml
See README.md and docs/known-limitations.md before running the research prototype.
"""

import base64
import io
import os
import sys
import json
import time
import tempfile
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import yaml
from fastapi import Request
from ray import serve
from ray.serve.handle import DeploymentHandle


# =========================
# Helpers
# =========================

def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Ray actors may run from a different working directory.
    for section in ("vision", "planner"):
        value = cfg.get(section, {}).get("system_prompt_path")
        if value:
            prompt_path = Path(value).expanduser()
            if not prompt_path.is_absolute():
                prompt_path = config_path.parent / prompt_path
            cfg[section]["system_prompt_path"] = str(prompt_path.resolve())
    return cfg

def load_system_prompt(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"System prompt file not found: {path}")
    _, ext = os.path.splitext(path)
    if ext == ".py":
        spec = importlib.util.spec_from_file_location("sys_prompt_mod", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        if not hasattr(mod, "SYSTEM_PROMPT"):
            raise AttributeError("Expected SYSTEM_PROMPT variable in the .py file.")
        return getattr(mod, "SYSTEM_PROMPT")
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

def pil_to_b64(img: Image.Image, fmt="JPEG", quality=90) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

def crop_pil(img: Image.Image, bbox: List[int]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    return img.crop((x1, y1, x2, y2))

def ensure_int_box(b):
    return [int(b[0]), int(b[1]), int(b[2]), int(b[3])]

# =========================
# YOLO video → per-frame detections
# =========================

class YoloVideoWrapper:
    def __init__(self, model_path: str, opts: Dict[str, Any]):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = float(opts.get("conf_threshold", 0.25))
        self.max_det = int(opts.get("max_det", 200))
        self.frame_stride = int(opts.get("frame_stride", 1))  # process every Nth frame

        # decoding control
        self.decode_with_opencv = bool(opts.get("decode_with_opencv", True))

    def _decode_video_with_opencv(self, video_bytes: bytes) -> List[Tuple[int, Image.Image]]:
        import cv2
        # OpenCV doesn't read from bytes directly; write a temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            tmp.write(video_bytes)
            tmp.flush()
            cap = cv2.VideoCapture(tmp.name)
            frames = []
            idx = 0
            ok, frame = cap.read()
            while ok:
                if idx % self.frame_stride == 0:
                    # BGR → RGB PIL
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    frames.append((idx, pil_img))
                idx += 1
                ok, frame = cap.read()
            cap.release()
        return frames

    def _maybe_decode_video(self, video_b64: str) -> List[Tuple[int, Image.Image]]:
        # If model can consume bytes directly, adapt here. Ultralytics model.predict on videos
        # typically expects a path/URI. We'll default to OpenCV decode so we can control stride.
        video_bytes = base64.b64decode(video_b64)
        if self.decode_with_opencv:
            return self._decode_video_with_opencv(video_bytes)
        # Fallback: still decode via OpenCV because most loaders need paths
        return self._decode_video_with_opencv(video_bytes)

    def infer(self, video_b64: str) -> Dict[str, Any]:
        """
        Returns:
        {
          "frames": [
            {
              "frame_index": int,
              "frame": "<base64>",
              "detections": [
                {"bbox":[x1,y1,x2,y2], "cls":"person", "score":0.9, "crop":"<b64>"},
                ...
              ]
            }, ...
          ]
        }
        """
        frames = self._maybe_decode_video(video_b64)  # list of (idx, PIL)

        out_frames: List[Dict[str, Any]] = []
        from ultralytics.utils.ops import scale_boxes  # used if needed

        for idx, pil_img in frames:
            np_img = np.array(pil_img)  # RGB
            results = self.model.predict(
                source=np_img,
                conf=self.conf,
                max_det=self.max_det,
                verbose=False
            )
            dets: List[Dict[str, Any]] = []
            for r in results:
                for b in r.boxes:
                    xyxy = ensure_int_box(b.xyxy[0].cpu().numpy())
                    cls_id = int(b.cls[0].item()) if b.cls is not None else -1
                    score = float(b.conf[0].item()) if b.conf is not None else 0.0
                    cls_name = r.names.get(cls_id, str(cls_id))
                    crop = crop_pil(pil_img, xyxy)
                    dets.append({
                        "bbox": xyxy,
                        "score": score,
                        "cls": cls_name,
                        "crop": pil_to_b64(crop)
                    })
            out_frames.append({
                "frame_index": idx,
                "frame": pil_to_b64(pil_img),
                "detections": dets
            })

        return {"frames": out_frames}

# =========================
# ByteTrack (multi-frame)
# =========================

class _NaiveMultiTracker:
    """Very simple multi-frame tracker: IoU matching per frame."""
    def __init__(self, iou_thresh=0.3, max_age=30):
        self.next_id = 1
        self.tracks = {}  # id -> bbox
        self.ages = {}    # id -> frames since last seen
        self.iou_thresh = iou_thresh
        self.max_age = max_age

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def update_sequence(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outputs = []
        last_frame_idx = -1
        for f in frames:
            idx = f["frame_index"]
            # age tracks once per new frame
            if idx != last_frame_idx:
                for tid in list(self.ages.keys()):
                    self.ages[tid] += 1
                    if self.ages[tid] > self.max_age:
                        self.tracks.pop(tid, None)
                        self.ages.pop(tid, None)
                last_frame_idx = idx

            dets = f.get("detections", [])
            assigned = set()
            tracked = []
            for d in dets:
                bb = d["bbox"]
                # best match existing
                best_id, best_iou = None, 0.0
                for tid, tbb in self.tracks.items():
                    i = self._iou(bb, tbb)
                    if i > best_iou:
                        best_iou, best_id = i, tid
                if best_iou >= self.iou_thresh and best_id is not None:
                    self.tracks[best_id] = bb
                    self.ages[best_id] = 0
                    tracked.append({**d, "track_id": best_id})
                    assigned.add(best_id)
                else:
                    tid = self.next_id; self.next_id += 1
                    self.tracks[tid] = bb
                    self.ages[tid] = 0
                    tracked.append({**d, "track_id": tid})
                    assigned.add(tid)

            outputs.append({
                "frame_index": idx,
                "frame": f["frame"],
                "tracks": tracked
            })
        return outputs


class ByteTrackVideoWrapper:
    # Helper class to wrap detection dicts in format expected by ultralytics BYTETracker
    class _DetectionsWrapper:
        """Wraps detection dicts to mimic ultralytics Results object interface"""
        def __init__(self, dets: List[Dict[str, Any]]):
            self.dets = dets
            # BYTETracker expects results.conf and results.xyxy
            # Extract and convert to numpy arrays
            self.conf = np.array([d.get("score", 0.5) for d in dets], dtype=np.float32)
            # xyxy format: [x1, y1, x2, y2]
            self.xyxy = np.array([d["bbox"] for d in dets], dtype=np.float32)
            
            # ByteTracker also needs cls (class IDs as integers)
            self.cls = np.zeros(len(dets), dtype=np.int32)
            
            # ByteTracker needs xywh format (center_x, center_y, width, height)
            if len(self.xyxy) > 0:
                x1, y1, x2, y2 = self.xyxy[:, 0], self.xyxy[:, 1], self.xyxy[:, 2], self.xyxy[:, 3]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                self.xywh = np.stack([cx, cy, w, h], axis=1)
            else:
                self.xywh = np.zeros((0, 4), dtype=np.float32)
        
        def __len__(self):
            return len(self.dets)
        
        def __getitem__(self, idx):
            """Support slicing like results[mask]"""
            if isinstance(idx, np.ndarray):
                # Boolean or integer indexing
                subset = [self.dets[i] for i in np.where(idx)[0]] if idx.dtype == bool else [self.dets[i] for i in idx]
                return ByteTrackVideoWrapper._DetectionsWrapper(subset)
            else:
                return ByteTrackVideoWrapper._DetectionsWrapper([self.dets[idx]])

    def __init__(self, opts: Dict[str, Any]):
        # Try Ultralytics BYTETracker; fallback to naive
        self.use_ultralytics = False
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
            class _Args:
                track_high_thresh = float(opts.get("track_high_thresh", 0.5))
                track_low_thresh = float(opts.get("track_low_thresh", 0.1))
                new_track_thresh = float(opts.get("new_track_thresh", 0.6))
                track_buffer = int(opts.get("track_buffer", 30))
                match_thresh = float(opts.get("match_thresh", 0.8))
                fuse_score = bool(opts.get("fuse_score", True))
                frame_rate = int(opts.get("frame_rate", 30))
                mot20 = bool(opts.get("mot20", False))
            self.bt = BYTETracker(_Args(), frame_rate=_Args.frame_rate)
            self.use_ultralytics = True
        except Exception:
            self.bt = _NaiveMultiTracker(
                iou_thresh=float(opts.get("match_thresh", 0.3)),
                max_age=int(opts.get("track_buffer", 30))
            )

    def _to_bytetrack_input(self, dets: List[Dict[str, Any]]) -> "_DetectionsWrapper":
        """Convert detection dicts to Results-like wrapper for BYTETracker"""
        return self._DetectionsWrapper(dets)

    def track_sequence(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.use_ultralytics:
            return self.bt.update_sequence(frames)

        # Ultralytics BYTETracker: update per frame
        tracked_frames: List[Dict[str, Any]] = []
        from ultralytics.trackers.byte_tracker import STrack  # for type hints

        # NOTE: BYTETracker.update expects a Results-like object with .conf and .xyxy attributes
        # We wrap our detections dict in _DetectionsWrapper to provide this interface
        im_hw = (1080, 1920)
        print(f"[ByteTrack] Starting tracking with {len(frames)} frames", file=sys.stderr)
        for f in frames:
            dets = f.get("detections", [])
            dets_results = self._to_bytetrack_input(dets)
            
            # BYTETracker.update returns list of STrack objects with tracking info
            try:
                online_targets = self.bt.update(dets_results, im_hw)
                print(f"[ByteTrack] Frame {f['frame_index']}: {len(dets)} detections -> {len(online_targets)} tracks", file=sys.stderr)
            except Exception as e:
                print(f"[ByteTrack] Unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                online_targets = []

            # build outputs; attach class/crop by nearest IoU from dets
            def iou(a, b):
                ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
                inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
                inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
                iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
                inter = iw * ih
                area_a = (ax2 - ax1) * (ay2 - ay1)
                area_b = (bx2 - bx1) * (by2 - by1)
                union = area_a + area_b - inter + 1e-6
                return inter / union

            tracks = []
            for t in online_targets:
                # ByteTracker returns numpy array: [x1, y1, x2, y2, track_id, score, cls, idx]
                x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                track_id = int(t[4])
                track_score = float(t[5])
                bbox = [x1, y1, x2, y2]
                # attach best det fields
                best, best_i = None, 0.0
                for d in dets:
                    ii = iou(bbox, d["bbox"])
                    if ii > best_i:
                        best_i, best = ii, d
                cls_name = best.get("cls", "object") if best else "object"
                crop_b64 = best.get("crop") if best else None
                tracks.append({
                    "track_id": track_id,
                    "bbox": bbox,
                    "cls": cls_name,
                    "score": track_score,
                    "crop": crop_b64
                })

            tracked_frames.append({
                "frame_index": f["frame_index"],
                "frame": f["frame"],
                "tracks": tracks
            })

        return tracked_frames


# =========================
# Qwen3-VL-8B (vision)
# =========================

class Qwen3VLSequenceWrapper:
    def __init__(self, model_id_or_path: str, opts: Dict[str, Any], system_prompt: str):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        import torch
        
        self.processor = AutoProcessor.from_pretrained(model_id_or_path)
        
        # Use flash_attention_2 if available for better performance
        attn_impl = opts.get("attn_implementation", "flash_attention_2")
        torch_dtype = opts.get("torch_dtype", torch.bfloat16)
        
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id_or_path,
                torch_dtype=torch_dtype,
                device_map=opts.get("device_map", "auto"),
                attn_implementation=attn_impl
            )
            print(f"[Qwen3VL] Loaded with {attn_impl}", file=sys.stderr)
        except Exception as e:
            print(f"[Qwen3VL] Flash attention failed, using default: {e}", file=sys.stderr)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id_or_path,
                torch_dtype=torch_dtype,
                device_map=opts.get("device_map", "auto")
            )
        
        self.max_new_tokens = int(opts.get("max_new_tokens", 256))
        self.system_prompt = system_prompt
        # No frame limit - process ALL frames to leverage Qwen3-VL's 256K context
        self.process_all_frames = True

    def _get_all_frames(self, tracked_frames: List[Dict[str, Any]]) -> List[Tuple[int, Image.Image]]:
        """
        Get ALL frames from the video sequence - no sampling.
        This leverages Qwen3-VL's 256K context window capability to process entire videos.
        """
        frames_by_idx = {f["frame_index"]: f for f in tracked_frames}
        pairs = []
        for f in tracked_frames:
            idx = f["frame_index"]
            b64 = f["frame"]
            pairs.append((idx, b64_to_pil(b64)))
        print(f"[Qwen3VL] Processing ALL {len(pairs)} frames (no sampling)", file=sys.stderr)
        return pairs

    def _build_text_context(self, tracked_frames: List[Dict[str, Any]]) -> str:
        # tiny summary: track ids trajectories
        traj = {}
        for f in tracked_frames:
            for t in f.get("tracks", []):
                tid = t["track_id"]
                bb = t["bbox"]
                traj.setdefault(tid, []).append((f["frame_index"], bb, t.get("cls", "object")))
        parts = []
        for tid, points in sorted(traj.items()):
            start = points[0][0]; end = points[-1][0]; cls = points[0][2]
            parts.append(f"ID {tid} ({cls}) from frame {start} to {end}")
        summary = "; ".join(parts) if parts else "No tracks."
        return f"Track summary: {summary}."

    def _infer_batched(self, all_frames: List[Tuple[int, Image.Image]], tracked_frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process video in batches and merge results - preserves ALL frames"""
        MAX_FRAMES_PER_BATCH = 50  # Reduced from 100 to help model generate better output
        num_batches = (len(all_frames) + MAX_FRAMES_PER_BATCH - 1) // MAX_FRAMES_PER_BATCH
        
        print(f"[Vision] 📦 Splitting {len(all_frames)} frames into {num_batches} batches", file=sys.stderr)
        print(f"[Vision] 📦 Splitting {len(all_frames)} frames into {num_batches} batches")  # Also to stdout
        
        all_thoughts = []
        all_critical = False
        max_confidence = 0
        descriptions = []
        goals = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * MAX_FRAMES_PER_BATCH
            end_idx = min((batch_idx + 1) * MAX_FRAMES_PER_BATCH, len(all_frames))
            batch_frames = all_frames[start_idx:end_idx]
            
            frame_range = f"frames {batch_frames[0][0]}-{batch_frames[-1][0]}"
            print(f"[Vision] Processing batch {batch_idx+1}/{num_batches}: {frame_range} ({len(batch_frames)} frames)", file=sys.stderr)
            print(f"[Vision] Processing batch {batch_idx+1}/{num_batches}: {frame_range} ({len(batch_frames)} frames)")  # Also to stdout
            
            # Process this batch
            batch_result = self._infer_single_batch(batch_frames, tracked_frames, batch_info=f"Batch {batch_idx+1}/{num_batches}, {frame_range}")
            
            print(f"[Vision] Batch {batch_idx+1} result keys: {list(batch_result.keys())}")  # Debug
            print(f"[Vision] Batch {batch_idx+1} thoughts count: {len(batch_result.get('thoughts', []))}")  # Debug
            
            # Merge results
            all_thoughts.extend(batch_result.get("thoughts", []))
            if batch_result.get("isCritical"):
                all_critical = True
            max_confidence = max(max_confidence, batch_result.get("confidence", 0))
            descriptions.append(batch_result.get("description", ""))
            goals.append(batch_result.get("goal", ""))
        
        # Combine results
        return {
            "qwen3_json": {
                "thoughts": all_thoughts,
                "verification": {
                    "value": True,
                    "reason": f"All {len(all_frames)} frames analyzed across {num_batches} batches with complete temporal coverage"
                },
                "isCritical": all_critical,
                "confidence": max_confidence,
                "description": " ".join(descriptions),
                "goal": goals[0] if goals else "Proceed with caution"
            }
        }

    def infer(self, tracked_frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Get ALL frames - no sampling
        all_frames = self._get_all_frames(tracked_frames)
        
        if not all_frames:
            return {"description": "No frames to analyze", "thoughts": [], 
                    "verification": {"value": True, "reason": "No data"}, 
                    "isCritical": False, "confidence": 0, "goal": "Proceed"}
        
        # BATCH PROCESSING: If too many frames, process in chunks and merge
        MAX_FRAMES_PER_BATCH = 50  # Safe limit per batch
        
        if len(all_frames) > MAX_FRAMES_PER_BATCH:
            print(f"[Vision] Processing {len(all_frames)} frames in batches of {MAX_FRAMES_PER_BATCH}", file=sys.stderr)
            return self._infer_batched(all_frames, tracked_frames)
        else:
            # Single batch processing - wrap the result
            result = self._infer_single_batch(all_frames, tracked_frames)
            
            # Debug: check what we got back
            with open(os.path.join(tempfile.gettempdir(), "qwen3_debug.txt"), "a", encoding="utf-8") as f:
                f.write(f"\n[Vision infer()] Got result from _infer_single_batch with keys: {list(result.keys())}\n")
                f.write(f"[Vision infer()] Description: {result.get('description', 'MISSING')[:100]}\n")
            
            wrapped = {"qwen3_json": result}
            with open(os.path.join(tempfile.gettempdir(), "qwen3_debug.txt"), "a", encoding="utf-8") as f:
                f.write(f"[Vision infer()] Wrapped result keys: {list(wrapped.keys())}\n")
                f.write(f"[Vision infer()] Wrapped['qwen3_json'] keys: {list(wrapped.get('qwen3_json', {}).keys())}\n")
            
            return wrapped

    def _infer_single_batch(self, all_frames: List[Tuple[int, Image.Image]], tracked_frames: List[Dict[str, Any]], batch_info: str = "") -> Dict[str, Any]:
        """Process a single batch of frames"""
        
        if not all_frames:
            return {"description": "No frames to analyze", "thoughts": [], 
                    "verification": {"value": True, "reason": "No data"}, 
                    "isCritical": False, "confidence": 0, "goal": "Proceed"}
        
        textual_context = self._build_text_context(tracked_frames)
        batch_note = f" ({batch_info})" if batch_info else ""

        # Build detailed prompt with system instructions and track context
        user_text = f"""You are analyzing a traffic scene from an autonomous vehicle camera{batch_note}.

{self.system_prompt}

Track Data:
{textual_context}

Video Information: Analyzing {len(all_frames)} frames sequentially from the complete video.

Based on ALL the images and track data above, provide a detailed JSON analysis following the exact format specified in the system prompt. Include:
- "thoughts" array with step_type, frame_range, object_code, thought, and validation for each object
- "verification" object with value (boolean) and reason
- "isCritical" (boolean) 
- "confidence" (0-100)
- "description" with detailed technical summary including object positions, trajectories, and spatial relationships across the entire video sequence
- "goal" with single imperative sentence for required action

Output ONLY valid JSON, no markdown, no extra text."""

        # Build messages in Qwen3-VL format with ALL frames
        # Qwen3-VL supports multiple images in a single message
        content = []
        
        # Add all frames as images
        for frame_idx, image in all_frames:
            content.append({"type": "image", "image": image})
        
        # Add text prompt after all images
        content.append({"type": "text", "text": user_text})
        
        messages = [{
            "role": "user",
            "content": content
        }]
        
        # Process with standard Qwen3-VL processor
        try:
            print(f"[Vision] Starting Qwen3-VL inference with {len(all_frames)} frames", file=sys.stderr)
            print(f"[Vision] First frame size: {all_frames[0][1].size}, mode: {all_frames[0][1].mode}", file=sys.stderr)
            
            # Apply chat template and tokenize
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate response with temperature for more diverse outputs
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.2
            )
            
            # Decode only the generated tokens (exclude input prompt)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            print(f"[Vision] Generated text (first 300 chars): {text[:300]}...", file=sys.stderr)
            
        except Exception as e:
            print(f"[Vision] Error in inference: {type(e).__name__}: {str(e)}", file=sys.stderr)
            print(f"[Vision] ❌ ERROR: {type(e).__name__}: {str(e)}")  # Also print to stdout
            import traceback
            traceback.print_exc(file=sys.stderr)
            traceback.print_exc()  # Also print to stdout
            # Fallback: simple JSON response
            text = json.dumps({
                "thoughts": [],
                "verification": {"value": False, "reason": "Inference failed"},
                "isCritical": False,
                "confidence": 0,
                "description": "Unable to analyze scene",
                "goal": "Proceed with caution"
            })

        # Parse JSON from output
        qwen_json = {}
        try:
            # Debug: write to file since print might not work
            with open(os.path.join(tempfile.gettempdir(), "qwen3_debug.txt"), "a", encoding="utf-8") as f:
                f.write(f"\n\n{'='*80}\n")
                f.write(f"[Vision Debug] Raw model output length: {len(text)} chars\n")
                f.write(f"[Vision Debug] Raw output:\n{text}\n")
                f.write(f"{'='*80}\n\n")
            
            # Also print
            print(f"[Vision] Raw model output length: {len(text)} chars")
            print(f"[Vision] Raw model output (first 500 chars): {text[:500]}")
            
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                print(f"[Vision] Found JSON match, length: {len(json_str)} chars")
                qwen_json = json.loads(json_str)
                print(f"[Vision] Successfully parsed JSON with keys: {list(qwen_json.keys())}")
            else:
                # If no JSON found, create minimal structure
                print(f"[Vision] ⚠️  No JSON found in model output!")
                with open(os.path.join(tempfile.gettempdir(), "qwen3_debug.txt"), "a", encoding="utf-8") as f:
                    f.write("[Vision Debug] NO JSON FOUND IN OUTPUT!\n")
                qwen_json = {
                    "thoughts": [],
                    "verification": {"value": False, "reason": "No JSON in response"},
                    "isCritical": False,
                    "confidence": 0,
                    "description": text[:200],
                    "goal": "Proceed safely"
                }
        except json.JSONDecodeError as e:
            print(f"[Vision] JSON parse error: {e}")
            print(f"[Vision] Failed JSON string (first 500 chars): {json_str[:500] if 'json_str' in locals() else 'N/A'}")
            qwen_json = {
                "thoughts": [],
                "verification": {"value": False, "reason": "JSON parse error"},
                "isCritical": False,
                "confidence": 0,
                "description": text[:200],
                "goal": "Proceed safely"
            }

        # Debug: write what we're returning
        with open(os.path.join(tempfile.gettempdir(), "qwen3_debug.txt"), "a", encoding="utf-8") as f:
            f.write(f"[Vision Debug] Returning qwen_json with keys: {list(qwen_json.keys())}\n")
            f.write(f"[Vision Debug] Description length: {len(qwen_json.get('description', ''))} chars\n")
            f.write(f"[Vision Debug] Thoughts count: {len(qwen_json.get('thoughts', []))}\n")
            f.write(f"[Vision Debug] Full return value:\n{json.dumps(qwen_json, indent=2)}\n")
        
        # Return unwrapped - caller will wrap if needed
        return qwen_json

# =========================
# Qwen planner
# =========================

class QwenPlannerWrapper:
    def __init__(self, model_id_or_path: str, opts: Dict[str, Any], planner_agent_prompt_path: str = None):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=opts.get("torch_dtype", "auto"),
            device_map=opts.get("device_map", "auto"),
            trust_remote_code=True
        )
        self.max_new_tokens = int(opts.get("max_new_tokens", 512))
        self.temperature = float(opts.get("temperature", 0.1))
        
        # Load planner agent prompt if provided
        self.planner_agent_prompt = None
        if planner_agent_prompt_path and os.path.exists(planner_agent_prompt_path):
            self.planner_agent_prompt = load_system_prompt(planner_agent_prompt_path)
            print(f"[QwenPlanner] Loaded planner agent prompt from {planner_agent_prompt_path}")

    def infer(self, description: str, qwen3_json: Dict[str, Any] = None) -> Dict[str, Any]:
        # Use planner agent prompt if available, otherwise use default
        if self.planner_agent_prompt:
            sys_prompt = self.planner_agent_prompt
        else:
            sys_prompt = (
                "You are a self-driving car planner. "
                "Using the provided scene description, output a BEHAVIOR TREE in strict JSON. "
                "Schema keys: {node_type, condition, action, params, children}. "
                "Return only JSON."
            )
        
        # Build user message with both description and Qwen3-VL JSON analysis
        user_parts = [f"Description:\n{description}"]
        
        if qwen3_json:
            user_parts.append(f"\n\nQwen3-VL Analysis (JSON):\n{json.dumps(qwen3_json, indent=2)}")
        
        user_parts.append("\n\nGenerate the behavior plan (initial behavior plan) as a JSON array of steps.")
        
        user = "".join(user_parts)
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ]
        input_ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        out_ids = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=self.temperature
        )
        
        # Decode ONLY the generated tokens (excluding input prompt)
        generated_tokens = out_ids[0][input_ids.shape[1]:]
        text = self.tok.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"[QwenPlanner] Generated response (first 300 chars): {text[:300]}", file=sys.stderr)

        # extract JSON
        import re
        m = re.search(r"\{.*\}", text, flags=re.S | re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                return {"behavior_plan": parsed}
            except Exception as e:
                print(f"[QwenPlanner] JSON parse error: {e}", file=sys.stderr)
        
        # If JSON extraction failed, look for array format
        m = re.search(r"\[.*\]", text, flags=re.S | re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                return {"behavior_plan": parsed}
            except Exception as e:
                print(f"[QwenPlanner] Array parse error: {e}", file=sys.stderr)
        
        return {"behavior_plan_raw": text}

# =========================
# Ray Serve deployments
# (memory is a scheduling hint, not a hard cap)
# =========================

@serve.deployment(
    name="yolo",
    num_replicas=1,
    ray_actor_options={
        "num_gpus": 0.25,     # use fractions if on a single GPU box
        "num_cpus": 4,
        "memory": 8 * 1024**3
    }
)
class YoloService:
    def __init__(self, model_path: str, opts: Dict[str, Any]):
        self.wrapper = YoloVideoWrapper(model_path, opts)

    async def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
        # Handle both dict (from Ray) and Request (from HTTP)
        if isinstance(request, dict):
            body = request
        else:
            body = await request.json()
        video_b64 = body["video"]
        return self.wrapper.infer(video_b64)


@serve.deployment(
    name="bytetrack",
    num_replicas=1,
    ray_actor_options={
        "num_gpus": 0.0,
        "num_cpus": 4,
        "memory": 4 * 1024**3
    }
)
class ByteTrackService:
    def __init__(self, opts: Dict[str, Any]):
        self.wrapper = ByteTrackVideoWrapper(opts)
    async def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
        # Handle both dict (from Ray) and Request (from HTTP)
        if isinstance(request, dict):
            body = request
        else:
            body = await request.json()
        tracked = self.wrapper.track_sequence(body["frames"])
        return {"frames": tracked}


@serve.deployment(
    name="vision",
    num_replicas=1,
    ray_actor_options={
        "num_gpus": 1.0,
        "num_cpus": 6,
        "memory": 20 * 1024**3
    }
)
class VisionService:
    def __init__(self, model_id_or_path: str, opts: Dict[str, Any], system_prompt_path: str):
        system_prompt = load_system_prompt(system_prompt_path)
        self.wrapper = Qwen3VLSequenceWrapper(model_id_or_path, opts, system_prompt)

    async def __call__(self, request: Dict[str, Any]) -> Dict[str, str]:
        # Handle both dict (from Ray) and Request (from HTTP)
        if isinstance(request, dict):
            body = request
        else:
            body = await request.json()
        return self.wrapper.infer(body["frames"])


@serve.deployment(
    name="planner",
    num_replicas=1,
    ray_actor_options={
        "num_gpus": 0.75,
        "num_cpus": 8,
        "memory": 16 * 1024**3
    }
)
class PlannerService:
    def __init__(self, model_id_or_path: str, opts: Dict[str, Any], planner_agent_prompt_path: str = None):
        self.wrapper = QwenPlannerWrapper(model_id_or_path, opts, planner_agent_prompt_path)

    async def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
        # Handle both dict (from Ray) and Request (from HTTP)
        if isinstance(request, dict):
            body = request
        else:
            body = await request.json()
        
        # Pass description and Qwen3-VL JSON to Qwen planner
        qwen3_json = body.get("qwen3_json", None)
        return self.wrapper.infer(body["description"], qwen3_json)


@serve.deployment(name="ingress", num_replicas=1)
class Ingress:
    """
    Request JSON:
      {
        "video": "<base64-encoded mp4/mov/avi>",
        "user_prompt": "... (optional)"
      }
    Response JSON: planner output (behavior tree)
    """
    def __init__(self,
                 yolo: DeploymentHandle,
                 bytetrack: DeploymentHandle,
                 vision: DeploymentHandle,
                 planner: DeploymentHandle):
        self.yolo = yolo
        self.bytetrack = bytetrack
        self.vision = vision
        self.planner = planner

    async def __call__(self, request: Request) -> Dict[str, Any]:
        try:
            pipeline_start = time.time()
            
            body = await request.json()
            video_b64 = body["video"]
            print(f"[Ingress] Received video of size {len(video_b64)} bytes")

            # 1) YOLO on all frames
            print("[Ingress] Starting YOLO detection...")
            yolo_start = time.time()
            yolo_out = await self.yolo.remote({"video": video_b64})
            yolo_time = time.time() - yolo_start
            print(f"[Ingress] YOLO completed: {len(yolo_out.get('frames', []))} frames in {yolo_time:.2f}s")
            
            # 2) ByteTrack across frames
            print("[Ingress] Starting ByteTrack...")
            bt_start = time.time()
            bt_out = await self.bytetrack.remote({"frames": yolo_out["frames"]})
            bt_time = time.time() - bt_start
            print(f"[Ingress] ByteTrack completed in {bt_time:.2f}s")
            
            # 3) Qwen3-VL vision over sequence → {qwen3_json}
            print("[Ingress] Starting Vision (Qwen3-VL)...")
            vis_start = time.time()
            vis_out = await self.vision.remote({"frames": bt_out["frames"]})
            vis_time = time.time() - vis_start
            print(f"[Ingress] Vision completed in {vis_time:.2f}s")
            
            # 4) Qwen planner with Qwen3-VL JSON analysis
            print("[Ingress] Starting Planner (Qwen)...")
            plan_start = time.time()
            qwen3_json = vis_out.get("qwen3_json", {})
            plan_out = await self.planner.remote({
                "description": qwen3_json.get("description", ""),
                "qwen3_json": qwen3_json
            })
            plan_time = time.time() - plan_start
            print(f"[Ingress] Planner completed in {plan_time:.2f}s")
            
            total_time = time.time() - pipeline_start
            
            print(f"\n{'='*80}")
            print(f"⏱️  PIPELINE TIMING SUMMARY:")
            print(f"   • YOLO Detection:    {yolo_time:.2f}s ({yolo_time/total_time*100:.1f}%)")
            print(f"   • ByteTrack:         {bt_time:.2f}s ({bt_time/total_time*100:.1f}%)")
            print(f"   • Qwen3-VL Vision:   {vis_time:.2f}s ({vis_time/total_time*100:.1f}%)")
            print(f"   • Qwen Planner:      {plan_time:.2f}s ({plan_time/total_time*100:.1f}%)")
            print(f"   • TOTAL:             {total_time:.2f}s")
            print(f"{'='*80}\n")
            
            # Return all outputs: YOLO, ByteTrack, Qwen3-VL vision, and Qwen planner
            return {
                "yolo_detection": {
                    "total_frames": len(yolo_out.get("frames", [])),
                    "frames": yolo_out.get("frames", [])
                },
                "bytetrack_tracking": {
                    "total_frames": len(bt_out.get("frames", [])),
                    "frames": bt_out.get("frames", [])
                },
                "qwen3_vision": qwen3_json,
                "qwen_planner": plan_out,
                "processing_time": {
                    "yolo_time": yolo_time,
                    "bytetrack_time": bt_time,
                    "qwen3_time": vis_time,
                    "qwen_time": plan_time,
                    "total_time": total_time
                }
            }
        except Exception as e:
            print(f"[Ingress] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return {"error": str(e), "type": type(e).__name__}


def main(config_path: str):
    import ray

    cfg = load_config(config_path)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Configure the HTTP proxy through Ray's public API.
    # An already running Serve instance retains its own HTTP options.
    serve.start(http_options={
        "host": cfg.get("serve", {}).get("host", "127.0.0.1"),
        "port": int(cfg.get("serve", {}).get("port", 8000)),
    })
    yolo_cfg, bt_cfg = cfg["yolo"], cfg["bytetrack"]
    vision_cfg, planner_cfg = cfg["vision"], cfg["planner"]
    yolo = YoloService.bind(model_path=yolo_cfg["model_path"], opts=yolo_cfg.get("opts", {}))
    bytetrack = ByteTrackService.bind(opts=bt_cfg.get("opts", {}))
    vision = VisionService.bind(
        model_id_or_path=vision_cfg["model"], opts=vision_cfg.get("opts", {}),
        system_prompt_path=vision_cfg["system_prompt_path"],
    )
    planner = PlannerService.bind(
        model_id_or_path=planner_cfg["model"], opts=planner_cfg.get("opts", {}),
        planner_agent_prompt_path=planner_cfg.get("system_prompt_path"),
    )
    app = Ingress.bind(yolo, bytetrack, vision, planner)
    serve.run(app, name="unexpected-situation", route_prefix="/")
    print("Application deployed. Press Ctrl+C to stop this application.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        serve.delete("unexpected-situation")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
