"""
FastAPI Backend — REST API + WebSocket for IBVAP.
Serves alerts, camera status, detection results, and live video streams.
"""
import cv2
import json
import time
import asyncio
import base64
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse
except ImportError:
    raise ImportError("fastapi not installed. Run: pip install fastapi uvicorn[standard]")

from ..edge.pipeline import IBVAPPipeline
from ..edge.hashchain import HashChain


def create_app(pipeline: Optional[IBVAPPipeline] = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="IBVAP API",
        description="Intelligent Border Video Analytics Platform",
        version="1.0.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # State
    if pipeline is None:
        pipeline = IBVAPPipeline()
        pipeline.load()
        pipeline.setup_fence()

    connected_clients: List[WebSocket] = []

    # ── REST Endpoints ──

    @app.get("/")
    async def root():
        return {"name": "IBVAP API", "version": "1.0.0", "status": "running"}

    @app.get("/api/status")
    async def get_status():
        """Get system status."""
        return pipeline.get_summary()

    @app.get("/api/alerts")
    async def get_alerts(last_n: int = 50):
        """Get recent alerts."""
        records = pipeline.hash_chain.get_records(last_n)
        return {"alerts": records, "total": len(pipeline.hash_chain)}

    @app.get("/api/alerts/verify")
    async def verify_chain():
        """Verify hash chain integrity."""
        is_valid, broken_at = pipeline.hash_chain.verify()
        return {
            "is_valid": is_valid,
            "broken_at": broken_at,
            "total_events": len(pipeline.hash_chain),
        }

    @app.get("/api/cameras")
    async def get_cameras():
        """Get camera status."""
        return pipeline.signal_detector.get_status()

    @app.get("/api/signal/thresholds")
    async def get_signal_thresholds():
        """Get current signal-loss detection thresholds."""
        return pipeline.signal_detector.get_thresholds()

    @app.post("/api/signal/thresholds")
    async def update_signal_thresholds(thresholds: dict):
        """Update signal-loss thresholds at runtime (for live demo tuning)."""
        import json, os
        config_path = os.path.join("config", "signal_thresholds.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(thresholds, f, indent=2)
        pipeline.signal_detector.reload_thresholds()
        return {"status": "ok", "thresholds": pipeline.signal_detector.get_thresholds()}

    @app.get("/api/fences")
    async def get_fences():
        """Get virtual fence zones."""
        return {"zones": pipeline.fence.get_all_zones()}

    @app.post("/api/fences")
    async def add_fence(zone: dict):
        """Add a virtual fence zone."""
        pipeline.fence.add_zone(
            name=zone["name"],
            polygon=zone["polygon"],
            severity=zone.get("severity", "high"),
            description=zone.get("description", ""),
        )
        return {"status": "ok", "zones": pipeline.fence.get_all_zones()}

    @app.delete("/api/fences/{zone_name}")
    async def remove_fence(zone_name: str):
        """Remove a virtual fence zone."""
        pipeline.fence.remove_zone(zone_name)
        return {"status": "ok"}

    @app.get("/api/tracks")
    async def get_tracks():
        """Get current tracked objects."""
        tracks = []
        for tid, traj in pipeline.tracker.track_history.items():
            if traj:
                tracks.append({
                    "track_id": tid,
                    "trajectory": traj[-20:],
                    "speed": pipeline.tracker.get_speed(tid),
                    "bearing": pipeline.tracker.get_bearing(tid),
                })
        return {"tracks": tracks}

    @app.get("/api/plates")
    async def get_plates():
        """Get ANPR results with per-frame readings and consensus.

        Each plate entry includes:
        - consensus: the majority-vote result (plate_text, confidence, num_frames, votes)
        - readings: list of individual per-frame OCR readings (for debugging/Q&A)
        """
        plates = []
        for tid, readings in pipeline.anpr.readings.items():
            consensus = pipeline.anpr.get_consensus(tid)
            if consensus:
                entry = consensus.to_dict()
                entry["readings"] = [r.to_dict() for r in readings]
                plates.append(entry)
        return {"plates": plates}

    @app.get("/api/chain")
    async def get_chain():
        """Get full hash chain."""
        return {"records": pipeline.hash_chain.get_all_records()}

    @app.get("/api/chain/export")
    async def export_chain():
        """Export hash chain as JSON."""
        return JSONResponse(
            content=json.loads(pipeline.hash_chain.export_json()),
            headers={"Content-Disposition": "attachment; filename=ibvap_chain.json"}
        )

    # ── Video Processing ──

    @app.post("/api/process/video")
    async def process_video(file: UploadFile = File(...), max_frames: int = 100):
        """Process an uploaded video file."""
        contents = await file.read()
        # Save temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        results = pipeline.process_video_file(tmp_path, max_frames=max_frames)

        import os
        os.unlink(tmp_path)

        return {
            "total_frames": len(results),
            "alerts": [a for r in results for a in r.alerts],
            "summary": pipeline.get_summary(),
        }

    # ── WebSocket for Live Streaming ──

    @app.websocket("/ws/live")
    async def websocket_live(websocket: WebSocket):
        """Live detection stream via WebSocket."""
        await websocket.accept()
        connected_clients.append(websocket)

        try:
            while True:
                # Receive frame as base64
                data = await websocket.receive_text()
                msg = json.loads(data)

                if msg.get("type") == "frame":
                    # Decode frame
                    img_bytes = base64.b64decode(msg["data"])
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        # Process frame
                        result = pipeline.process_frame(frame)

                        # Send back annotated frame + results
                        if result.annotated_frame is not None:
                            _, buffer = cv2.imencode('.jpg', result.annotated_frame,
                                                      [cv2.IMWRITE_JPEG_QUALITY, 80])
                            img_b64 = base64.b64encode(buffer).decode()

                            await websocket.send_json({
                                "type": "result",
                                "frame": img_b64,
                                "frame_id": result.frame_id,
                                "detections": result.tracks,
                                "alerts": result.alerts,
                                "intrusions": result.intrusions,
                                "plates": result.plates,
                                "chain_head": result.hash_chain_head,
                            })

                elif msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

        except WebSocketDisconnect:
            connected_clients.remove(websocket)

    # ── Broadcast helper ──

    async def broadcast_alert(alert: dict):
        """Broadcast alert to all connected WebSocket clients."""
        message = json.dumps({"type": "alert", "data": alert})
        for client in connected_clients[:]:
            try:
                await client.send_text(message)
            except Exception:
                connected_clients.remove(client)

    # Register alert broadcaster
    pipeline.on_alert(lambda record: asyncio.create_task(
        broadcast_alert(record.to_dict())
    ) if connected_clients else None)

    return app
