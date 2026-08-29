"""
IBVAP — Main Entry Point
Run: python main.py
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI backend server."""
    import uvicorn
    from src.backend.api import create_app

    app = create_app()
    print(f"🚀 IBVAP API running on http://{host}:{port}")
    print(f"📖 Docs: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)


def run_dashboard():
    """Run the Streamlit dashboard."""
    import subprocess
    print("🛡️  Starting IBVAP Dashboard...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "web_demo.py",
        "--server.port", "8501",
        "--server.headless", "true",
    ])


def run_demo(video_path: str = None):
    """Run detection on a video file or camera."""
    import cv2
    from src.edge.pipeline import IBVAPPipeline

    pipeline = IBVAPPipeline()
    pipeline.load()
    pipeline.setup_fence()

    print("🛡️  IBVAP Pipeline initialized")
    print(f"   Model: yolov8n.pt")
    print(f"   Fence zones: {list(pipeline.fence.zones.keys())}")
    print()

    if video_path:
        print(f"📹 Processing: {video_path}")
        cap = cv2.VideoCapture(video_path)
    else:
        print("📷 Using webcam...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open video source")
        return

    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = pipeline.process_frame(frame)
            frame_count += 1

            if result.annotated_frame is not None:
                cv2.imshow("IBVAP", result.annotated_frame)

            if result.alerts:
                for alert in result.alerts:
                    print(f"  🚨 [{alert['severity'].upper()}] {alert['event_type']}: "
                          f"{alert.get('explanation', '')}")

            if frame_count % 100 == 0:
                summary = pipeline.get_summary()
                print(f"  📊 Frame {frame_count} | "
                      f"Alerts: {summary['total_alerts']} | "
                      f"Chain: {'✅' if summary['chain_valid'] else '❌'}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n⏹ Stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Print summary
        summary = pipeline.get_summary()
        print(f"\n📊 Session Summary:")
        print(f"   Frames processed: {summary['total_frames']}")
        print(f"   Total alerts: {summary['total_alerts']}")
        print(f"   Chain valid: {summary['chain_valid']}")
        print(f"   Chain head: {summary['head_hash']}")


def main():
    parser = argparse.ArgumentParser(description="IBVAP - Border Video Analytics Platform")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Server
    server_parser = subparsers.add_parser("server", help="Run API server")
    server_parser.add_argument("--host", default="0.0.0.0")
    server_parser.add_argument("--port", type=int, default=8000)

    # Dashboard
    subparsers.add_parser("dashboard", help="Run Streamlit dashboard")

    # Demo
    demo_parser = subparsers.add_parser("demo", help="Run demo on video/camera")
    demo_parser.add_argument("--video", help="Path to video file")

    args = parser.parse_args()

    if args.command == "server":
        run_server(args.host, args.port)
    elif args.command == "dashboard":
        run_dashboard()
    elif args.command == "demo":
        run_demo(args.video)
    else:
        # Default: run dashboard
        run_dashboard()


if __name__ == "__main__":
    main()
