"""
IBVAP Backend API
FastAPI server for event ingestion, alert management, and dashboard
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import json
import asyncio
from collections import defaultdict
import hashlib


# ============================================================
# Data Models
# ============================================================

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventType(str, Enum):
    FENCE_INTRUSION = "fence_intrusion"
    VEHICLE_DETECTED = "vehicle_detected"
    PERSON_DETECTED = "person_detected"
    ANPR_MATCH = "anpr_match"
    SIGNAL_LOSS = "signal_loss"
    BEHAVIORAL_DEVIATION = "behavioral_deviation"


class AlertEvent(BaseModel):
    """Tamper-evident alert event"""
    event_id: str = Field(..., description="Unique event identifier")
    prev_hash: str = Field(..., description="Hash of previous event for chain integrity")
    timestamp: datetime = Field(..., description="Event timestamp in ISO format")
    site_id: str = Field(..., description="Border outpost identifier")
    camera_id: str = Field(..., description="Camera identifier")
    event_type: EventType = Field(..., description="Type of event")
    object_class: str = Field(..., description="Detected object class (person/vehicle/none)")
    track_id: str = Field(..., description="Persistent track identifier")
    zone: str = Field(..., description="Zone name where event occurred")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    explanation: str = Field(..., description="Human-readable explanation")
    severity: SeverityLevel = Field(..., description="Alert severity level")
    clip_ref: str = Field(..., description="Reference to video clip")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "e7f1a2b3c4d5e6f7",
                "prev_hash": "a92c3d4e5f6a7b8c",
                "timestamp": "2026-08-29T21:14:03Z",
                "site_id": "BOP-14",
                "camera_id": "CAM-2",
                "event_type": "fence_intrusion",
                "object_class": "person",
                "track_id": "T-0042",
                "zone": "Zone-3",
                "confidence": 0.91,
                "explanation": "Track T-0042 crossed virtual fence Zone-3 at 1.4 m/s, bearing NE",
                "severity": "high",
                "clip_ref": "s3://ibvap-clips/e7f1a2b3c4d5e6f7.mp4"
            }
        }


class SiteStatus(BaseModel):
    """Site status information"""
    site_id: str
    site_name: str
    latitude: float
    longitude: float
    cameras: List[Dict[str, Any]]
    last_event: Optional[datetime]
    alert_count: int
    status: str  # online, offline, degraded


class DashboardStats(BaseModel):
    """Dashboard statistics"""
    total_sites: int
    online_sites: int
    total_cameras: int
    online_cameras: int
    total_alerts_today: int
    critical_alerts: int
    high_alerts: int
    avg_response_time_ms: float


# ============================================================
# Application
# ============================================================

app = FastAPI(
    title="IBVAP API",
    description="Intelligent Border Video Analytics Platform API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (replace with PostgreSQL in production)
events_db: List[AlertEvent] = []
sites_db: Dict[str, SiteStatus] = {}
websocket_connections: List[WebSocket] = []


# ============================================================
# API Endpoints
# ============================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "name": "IBVAP API",
        "version": "1.0.0",
        "description": "Intelligent Border Video Analytics Platform",
        "docs": "/docs"
    }


@app.post("/api/v1/events", response_model=AlertEvent, tags=["Events"])
async def ingest_event(event: AlertEvent):
    """
    Ingest a new alert event from edge nodes
    
    This endpoint receives events from edge detection modules
    and stores them in the event database.
    """
    # Verify hash chain
    if events_db:
        last_event = events_db[-1]
        if event.prev_hash != hashlib.sha256(
            json.dumps(last_event.model_dump(), sort_keys=True).encode()
        ).hexdigest():
            raise HTTPException(
                status_code=400,
                detail="Hash chain verification failed"
            )
    
    # Store event
    events_db.append(event)
    
    # Broadcast to WebSocket clients
    await broadcast_event(event)
    
    return event


@app.get("/api/v1/events", response_model=List[AlertEvent], tags=["Events"])
async def get_events(
    site_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    event_type: Optional[EventType] = None,
    severity: Optional[SeverityLevel] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    Get events with optional filtering
    """
    filtered = events_db
    
    if site_id:
        filtered = [e for e in filtered if e.site_id == site_id]
    if camera_id:
        filtered = [e for e in filtered if e.camera_id == camera_id]
    if event_type:
        filtered = [e for e in filtered if e.event_type == event_type]
    if severity:
        filtered = [e for e in filtered if e.severity == severity]
    
    # Apply pagination
    return filtered[-limit - offset:-offset] if offset else filtered[-limit:]


@app.get("/api/v1/events/{event_id}", response_model=AlertEvent, tags=["Events"])
async def get_event(event_id: str):
    """Get a specific event by ID"""
    for event in events_db:
        if event.event_id == event_id:
            return event
    raise HTTPException(status_code=404, detail="Event not found")


@app.post("/api/v1/events/verify-chain", tags=["Events"])
async def verify_hash_chain():
    """
    Verify the integrity of the event hash chain
    
    Returns whether the chain is valid and the index of any tampered event
    """
    if not events_db:
        return {"valid": True, "tampered_index": None, "total_events": 0}
    
    for i in range(1, len(events_db)):
        prev_event = events_db[i - 1].model_dump()
        prev_event.pop("prev_hash", None)
        
        calculated_hash = hashlib.sha256(
            json.dumps(prev_event, sort_keys=True).encode()
        ).hexdigest()
        
        if calculated_hash != events_db[i].prev_hash:
            return {
                "valid": False,
                "tampered_index": i,
                "total_events": len(events_db)
            }
    
    return {"valid": True, "tampered_index": None, "total_events": len(events_db)}


@app.get("/api/v1/sites", response_model=List[SiteStatus], tags=["Sites"])
async def get_sites():
    """Get all border outpost sites"""
    return list(sites_db.values())


@app.get("/api/v1/sites/{site_id}", response_model=SiteStatus, tags=["Sites"])
async def get_site(site_id: str):
    """Get a specific site"""
    if site_id in sites_db:
        return sites_db[site_id]
    raise HTTPException(status_code=404, detail="Site not found")


@app.get("/api/v1/stats", response_model=DashboardStats, tags=["Stats"])
async def get_dashboard_stats():
    """Get dashboard statistics"""
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    today_events = [e for e in events_db if e.timestamp >= today_start]
    
    return DashboardStats(
        total_sites=len(sites_db),
        online_sites=sum(1 for s in sites_db.values() if s.status == "online"),
        total_cameras=sum(len(s.cameras) for s in sites_db.values()),
        online_cameras=sum(
            sum(1 for c in s.cameras if c.get("status") == "online")
            for s in sites_db.values()
        ),
        total_alerts_today=len(today_events),
        critical_alerts=sum(1 for e in today_events if e.severity == SeverityLevel.CRITICAL),
        high_alerts=sum(1 for e in today_events if e.severity == SeverityLevel.HIGH),
        avg_response_time_ms=150.0  # Simulated
    )


# ============================================================
# WebSocket for Real-time Updates
# ============================================================

@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """
    WebSocket endpoint for real-time event streaming
    
    Clients connected to this endpoint receive events in real-time
    as they are ingested from edge nodes.
    """
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            
            # Echo back or handle commands
            if data == "ping":
                await websocket.send_text("pong")
            
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)


async def broadcast_event(event: AlertEvent):
    """Broadcast event to all connected WebSocket clients"""
    message = json.dumps({
        "type": "event",
        "data": event.model_dump()
    })
    
    disconnected = []
    for connection in websocket_connections:
        try:
            await connection.send_text(message)
        except Exception:
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        websocket_connections.remove(conn)


# ============================================================
# MQTT Integration (for C2 systems)
# ============================================================

class MQTTIntegration:
    """
    MQTT integration for C2 system communication
    
    Publishes events to MQTT topics for consumption by
    command and control systems.
    """
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.connected = False
        
        print(f"[MQTT] Would connect to {broker_host}:{broker_port}")
    
    def publish_event(self, event: AlertEvent):
        """
        Publish event to MQTT topic
        
        Topic structure: ibvap/{site_id}/{camera_id}/{event_type}
        """
        topic = f"ibvap/{event.site_id}/{event.camera_id}/{event.event_type.value}"
        payload = json.dumps(event.model_dump(), default=str)
        
        print(f"[MQTT] Would publish to {topic}")
        print(f"[MQTT] Payload: {payload[:100]}...")
        
        # In production, use paho-mqtt:
        # import paho.mqtt.client as mqtt
        # client.publish(topic, payload)


# Initialize MQTT integration
mqtt_integration = MQTTIntegration()


# ============================================================
# Demo Data
# ============================================================

def populate_demo_data():
    """Populate database with demo data"""
    global sites_db
    
    # Add demo sites
    sites_db["BOP-01"] = SiteStatus(
        site_id="BOP-01",
        site_name="Border Post Alpha",
        latitude=34.0837,
        longitude=74.7973,
        cameras=[
            {"camera_id": "CAM-01", "status": "online", "type": "fixed"},
            {"camera_id": "CAM-02", "status": "online", "type": "ptz"},
            {"camera_id": "CAM-03", "status": "offline", "type": "fixed"}
        ],
        last_event=datetime.utcnow(),
        alert_count=12,
        status="online"
    )
    
    sites_db["BOP-02"] = SiteStatus(
        site_id="BOP-02",
        site_name="Border Post Bravo",
        latitude=34.1234,
        longitude=74.8123,
        cameras=[
            {"camera_id": "CAM-01", "status": "online", "type": "fixed"},
            {"camera_id": "CAM-02", "status": "online", "type": "fixed"}
        ],
        last_event=datetime.utcnow(),
        alert_count=5,
        status="online"
    )
    
    # Add demo events
    for i in range(10):
        event = AlertEvent(
            event_id=f"event-{i:04d}",
            prev_hash=hashlib.sha256(f"prev-{i}".encode()).hexdigest()[:64],
            timestamp=datetime.utcnow(),
            site_id="BOP-01",
            camera_id=f"CAM-0{(i % 3) + 1}",
            event_type=EventType.FENCE_INTRUSION if i % 2 == 0 else EventType.VEHICLE_DETECTED,
            object_class="person" if i % 2 == 0 else "vehicle",
            track_id=f"T-{i:04d}",
            zone=f"Zone-{(i % 3) + 1}",
            confidence=0.85 + (i % 10) * 0.01,
            explanation=f"Track T-{i:04d} detected in Zone-{(i % 3) + 1}",
            severity=SeverityLevel.HIGH if i % 3 == 0 else SeverityLevel.MEDIUM,
            clip_ref=f"s3://ibvap-clips/event-{i:04d}.mp4"
        )
        events_db.append(event)


# ============================================================
# Startup/Shutdown Events
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("=" * 60)
    print("IBVAP Backend Starting...")
    print("=" * 60)
    populate_demo_data()
    print(f"Loaded {len(sites_db)} sites")
    print(f"Loaded {len(events_db)} events")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("IBVAP Backend Shutting Down...")


# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("Starting IBVAP Backend Server...")
    print("API docs available at: http://localhost:8000/docs")
    print("ReDoc available at: http://localhost:8000/redoc")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
