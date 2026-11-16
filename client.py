# client_send_video.py
import argparse
import base64
import json
import os
import sys
import time
import requests

def encode_video_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def send_video(server_url: str, video_b64: str, user_prompt: str = "", timeout: int = 600):
    payload = {"video": video_b64}
    if user_prompt:
        payload["user_prompt"] = user_prompt
    
    print(f"Payload size: {len(json.dumps(payload)) / (1024*1024):.2f} MB")
    
    try:
        r = requests.post(server_url.rstrip("/") + "/", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError as e:
        print(f"Connection error: {e}", file=sys.stderr)
        print(f"Make sure the SSH tunnel is running and the remote server is active", file=sys.stderr)
        print(f"Server URL being used: {server_url}", file=sys.stderr)
        raise

def main():
    ap = argparse.ArgumentParser(description="Send a video to the Ray Serve pipeline and save outputs.")
    ap.add_argument("video", help="Path to video file (e.g., .mp4)")
    ap.add_argument("--server", default="http://localhost:8000", help="Ingress base URL (default: http://localhost:8000)")
    ap.add_argument("--prompt", default="", help="Optional extra user prompt")
    ap.add_argument("--timeout", type=int, default=600, help="Request timeout seconds (default: 600)")
    ap.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Encoding video: {args.video} ...")
    encode_start = time.time()
    video_b64 = encode_video_b64(args.video)
    encode_time = time.time() - encode_start
    print(f"✅ Video encoded in {encode_time:.2f} seconds")

    print(f"\nSending to {args.server}/ ...")
    request_start = time.time()
    try:
        resp = send_video(args.server, video_b64, user_prompt=args.prompt, timeout=args.timeout)
        request_time = time.time() - request_start
    except requests.HTTPError as e:
        print(f"HTTP error: {e}\nResponse: {getattr(e, 'response', None) and e.response.text}", file=sys.stderr)
        sys.exit(2)
    except requests.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(3)

    print(f"\n{'='*80}")
    print(f"⏱️  TOTAL PROCESSING TIME: {request_time:.2f} seconds")
    print(f"{'='*80}\n")
    
    # Extract timing information from response if available
    if "processing_time" in resp:
        timing = resp["processing_time"]
        print(f"📊 STAGE BREAKDOWN:")
        if "yolo_time" in timing:
            print(f"   • YOLO Detection:    {timing['yolo_time']:.2f}s")
        if "bytetrack_time" in timing:
            print(f"   • ByteTrack:         {timing['bytetrack_time']:.2f}s")
        if "qwen3_time" in timing:
            print(f"   • Qwen3-VL Vision:   {timing['qwen3_time']:.2f}s")
        if "qwen_time" in timing:
            print(f"   • Qwen Planner:      {timing['qwen_time']:.2f}s")
        if "total_time" in timing:
            print(f"   • Pipeline Total:    {timing['total_time']:.2f}s")
        print()

    # Generate base filename from video name
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    
    # Save YOLO detection output (with frames)
    yolo_output_file = os.path.join(args.output_dir, f"{video_name}_yolo_detection.json")
    with open(yolo_output_file, "w", encoding="utf-8") as f:
        json.dump(resp.get("yolo_detection", {}), f, indent=2, ensure_ascii=False)
    print(f"✅ YOLO detection output saved to: {yolo_output_file}")
    
    # Save ByteTrack tracking output (with frames)
    bytetrack_output_file = os.path.join(args.output_dir, f"{video_name}_bytetrack_tracking.json")
    with open(bytetrack_output_file, "w", encoding="utf-8") as f:
        json.dump(resp.get("bytetrack_tracking", {}), f, indent=2, ensure_ascii=False)
    print(f"✅ ByteTrack tracking output saved to: {bytetrack_output_file}")
    
    # Save Qwen3-VL vision output
    qwen3_output_file = os.path.join(args.output_dir, f"{video_name}_qwen3_vision.json")
    with open(qwen3_output_file, "w", encoding="utf-8") as f:
        json.dump(resp.get("qwen3_vision", {}), f, indent=2, ensure_ascii=False)
    print(f"✅ Qwen3-VL output saved to: {qwen3_output_file}")
    
    # Save Qwen planner output
    qwen_output_file = os.path.join(args.output_dir, f"{video_name}_qwen_planner.json")
    with open(qwen_output_file, "w", encoding="utf-8") as f:
        json.dump(resp.get("qwen_planner", {}), f, indent=2, ensure_ascii=False)
    print(f"✅ Qwen planner output saved to: {qwen_output_file}")
    
    # Save combined output summary
    combined_output_file = os.path.join(args.output_dir, f"{video_name}_complete_output.json")
    combined = {
        "yolo_detection": {"total_frames": resp.get("yolo_detection", {}).get("total_frames")},
        "bytetrack_tracking": {"total_frames": resp.get("bytetrack_tracking", {}).get("total_frames")},
        "qwen3_vision": resp.get("qwen3_vision", {}),
        "qwen_planner": resp.get("qwen_planner", {})
    }
    with open(combined_output_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"✅ Complete output summary saved to: {combined_output_file}")
    
    print(f"\n📁 All files saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()