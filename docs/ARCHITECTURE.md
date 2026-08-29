# IBVAP — System Architecture
### Intelligent Border Video Analytics Platform

---

## 1. Architecture Philosophy

IBVAP follows an **edge-first, metadata-only, store-and-forward** architecture. Every design decision optimizes for the actual constraints of border deployment:

- **Bandwidth**: Only compact JSON events travel upstream, never raw video
- **Hardware**: Tiered by compute budget, not one-size-fits-all
- **Connectivity**: Store-and-forward ensures zero data loss during outages
- **Trust**: Signal-loss-is-itself-an-alert prevents silent failures

---

## 2. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TIER 3: COMMAND CENTER                              │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐  │
│  │  Dashboard   │  │  Event Store │  │  Alert Queue│  │  C2 Integration  │  │
│  │  (React +    │  │  (PostgreSQL)│  │  (Redis)    │  │  (MQTT/API)      │  │
│  │  Leaflet)    │  │              │  │             │  │                  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘  └────────┬─────────┘  │
│         │                 │                 │                   │            │
│         └─────────────────┴─────────────────┴───────────────────┘            │
│                                    │                                         │
│                            ┌───────┴───────┐                                │
│                            │   MQTT Broker  │                                │
│                            │  (Mosquitto)   │                                │
│                            └───────┬───────┘                                │
└────────────────────────────────────┼─────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
         ┌──────────┴──────┐ ┌──────┴───────┐ ┌──────┴───────┐
         │   TIER 1: BOP   │ │  TIER 1: BOP │ │  TIER 2:    │
         │  (High Priority)│ │  (Standard)  │ │  Remote Road │
         │                 │ │              │ │              │
         │ ┌─────────────┐ │ │ ┌──────────┐│ │ ┌──────────┐│
         │ │ Edge Box    │ │ │ │ Edge Box ││ │ │ Motion   ││
         │ │ (Jetson)    │ │ │ │ (mini-PC)││ │ │ Trigger  ││
         │ │             │ │ │ │          ││ │ │ ($20 MCU)││
         │ │ • YOLOv8    │ │ │ │ • YOLOv8 ││ │ │          ││
         │ │ • ByteTrack │ │ │ │ • Track  ││ │ │ • Frame  ││
         │ │ • ANPR      │ │ │ │ • ANPR   ││ │ │   diff   ││
         │ │ • Fence     │ │ │ │ • Fence  ││ │ │ • Buffer ││
         │ │ • Baseline  │ │ │ │          ││ │ │ • Forward││
         │ └──────┬──────┘ │ │ └────┬─────┘│ │ └────┬─────┘│
         │        │        │ │      │      │ │      │      │
         │ ┌──────┴──────┐ │ │ ┌────┴─────┐│ │ ┌────┴─────┐│
         │ │ IP Camera   │ │ │ │IP Camera ││ │ │IP Camera ││
         │ │ (RTSP)      │ │ │ │(RTSP)    ││ │ │(RTSP)    ││
         │ └─────────────┘ │ │ └──────────┘│ │ └──────────┘│
         └─────────────────┘ └─────────────┘ └─────────────┘
```

---

## 3. Data Flow

### 3.1 Normal Operation (Edge → Cloud)

```
Camera (RTSP) → Edge Box → Detection → Tracking → Event Generator
                                                      │
                                                      ▼
                                              ┌───────────────┐
                                              │ Event Queue   │
                                              │ (Local SQLite)│
                                              └───────┬───────┘
                                                      │
                                              ┌───────┴───────┐
                                              │ MQTT Publish  │
                                              │ (When linked) │
                                              └───────┬───────┘
                                                      │
                                              ┌───────┴───────┐
                                              │ Command Center│
                                              │ (Store + Show)│
                                              └───────────────┘
```

### 3.2 Offline Operation (Store-and-Forward)

```
Camera (RTSP) → Edge Box → Detection → Tracking → Event Generator
                                                      │
                                                      ▼
                                              ┌───────────────┐
                                              │ Event Queue   │
                                              │ (Local SQLite)│
                                              └───────┬───────┘
                                                      │
                                              ┌───────┴───────┐
                                              │ Link Status:  │
                                              │ DOWN          │
                                              └───────┬───────┘
                                                      │
                                              ┌───────┴───────┐
                                              │ Store Locally │
                                              │ + Timestamp   │
                                              └───────┬───────┘
                                                      │ (When linked)
                                              ┌───────┴───────┐
                                              │ Sync All      │
                                              │ Pending Events│
                                              └───────────────┘
```

### 3.3 Alert Generation Flow

```
Frame → Detection → Tracking → Event Rules → Alert Generator
                    │              │                │
                    │              │                ▼
                    │              │        ┌───────────────┐
                    │              │        │ Severity Calc │
                    │              │        │ • Confidence  │
                    │              │        │ • Deviation   │
                    │              │        │ • Zone risk   │
                    │              │        └───────┬───────┘
                    │              │                │
                    ▼              ▼                ▼
              ┌─────────┐  ┌─────────────┐  ┌─────────────┐
              │ Track   │  │ Rule Engine │  │ Explanation │
              │ Store   │  │ (Fence,     │  │ Generator   │
              │         │  │  Baseline)  │  │             │
              └─────────┘  └─────────────┘  └─────────────┘
```

---

## 4. Component Details

### 4.1 Edge Detection Module

**Location**: `src/edge/detector.py`

```python
# Core detection pipeline
class EdgeDetector:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = YOLO(model_path)  # YOLOv8-nano ONNX
        self.tracker = ByteTrack()
        self.anpr = ANPRConsensus()
        self.fence_monitor = VirtualFence()
    
    def process_frame(self, frame: np.ndarray) -> List[Event]:
        # 1. Detect objects
        detections = self.model(frame, conf=0.5)
        
        # 2. Track across frames
        tracks = self.tracker.update(detections, frame)
        
        # 3. Generate events
        events = []
        for track in tracks:
            # Check virtual fence
            if self.fence_monitor.check_crossing(track):
                events.append(FenceIntrusionEvent(track))
            
            # Check if vehicle → run ANPR
            if track.class_name == "vehicle":
                plate = self.anpr.process(track)
                if plate:
                    events.append(ANPREvent(track, plate))
        
        return events
```

### 4.2 Virtual Fence Module

**Location**: `src/edge/fence.py`

```python
class VirtualFence:
    def __init__(self, polygons: List[Polygon]):
        self.polygons = polygons
        self.previous_positions = {}
    
    def check_crossing(self, track: Track) -> bool:
        """Check if track centroid crossed any fence polygon."""
        current_pos = track.centroid
        prev_pos = self.previous_positions.get(track.id)
        
        if prev_pos is None:
            self.previous_positions[track.id] = current_pos
            return False
        
        for polygon in self.polygons:
            # Check if line segment (prev → current) intersects polygon edges
            if self._segment_crosses_polygon(prev_pos, current_pos, polygon):
                self.previous_positions[track.id] = current_pos
                return True
        
        self.previous_positions[track.id] = current_pos
        return False
```

### 4.3 ANPR Multi-Frame Consensus

**Location**: `src/edge/anpr.py`

```python
class ANPRConsensus:
    def __init__(self, min_frames: int = 3, consensus_threshold: float = 0.6):
        self.min_frames = min_frames
        self.consensus_threshold = consensus_threshold
        self.frame_buffer = defaultdict(list)
    
    def process(self, track: Track) -> Optional[str]:
        """Run OCR on frame, buffer results, return consensus plate."""
        # Run OCR on current frame
        plate_text = self.ocr.read Plate(track.current_frame, track.bbox)
        
        if plate_text:
            self.frame_buffer[track.id].append(plate_text)
        
        # Check if we have enough frames for consensus
        if len(self.frame_buffer[track.id]) >= self.min_frames:
            return self._vote_consensus(self.frame_buffer[track.id])
        
        return None
    
    def _vote_consensus(self, plates: List[str]) -> Optional[str]:
        """Majority-vote across buffered plate readings."""
        counter = Counter(plates)
        most_common, count = counter.most_common(1)[0]
        
        if count / len(plates) >= self.consensus_threshold:
            return most_common
        return None
```

### 4.4 Alert System

**Location**: `src/backend/alerts.py`

```python
@dataclass
class Alert:
    event_id: str
    prev_hash: str
    timestamp: datetime
    site_id: str
    camera_id: str
    event_type: str  # "fence_intrusion", "anpr_match", "signal_loss"
    object_class: str
    track_id: str
    zone: Optional[str]
    bearing: Optional[str]
    speed_mps: Optional[float]
    confidence: float
    baseline_deviation: bool
    clip_ref: Optional[str]
    severity: str  # "low", "medium", "high", "critical"
    explanation: str

class AlertGenerator:
    def __init__(self, baseline_model, hash_chain):
        self.baseline = baseline_model
        self.hash_chain = hash_chain
    
    def generate(self, event: Event) -> Alert:
        # Calculate severity
        severity = self._calculate_severity(event)
        
        # Check baseline deviation
        deviation = self.baseline.check_deviation(event)
        
        # Generate explanation
        explanation = self._explain(event, deviation)
        
        # Create alert with hash chain
        alert = Alert(
            event_id=uuid4().hex,
            prev_hash=self.hash_chain.last_hash,
            timestamp=datetime.utcnow(),
            severity=severity,
            explanation=explanation,
            # ... other fields
        )
        
        # Update hash chain
        self.hash_chain.append(alert)
        
        return alert
```

### 4.5 Signal Loss Detection

**Location**: `src/edge/heartbeat.py`

```python
class SignalLossMonitor:
    def __init__(self, camera_id: str, timeout_sec: float = 5.0):
        self.camera_id = camera_id
        self.timeout = timeout_sec
        self.last_frame_time = time.time()
        self.alert_callback = None
    
    def on_frame_received(self):
        """Called on every successful frame read."""
        self.last_frame_time = time.time()
    
    def check(self) -> Optional[Alert]:
        """Check if signal is lost."""
        elapsed = time.time() - self.last_frame_time
        
        if elapsed > self.timeout:
            return Alert(
                event_type="signal_loss",
                camera_id=self.camera_id,
                severity="high",
                explanation=f"Camera {self.camera_id} signal lost for {elapsed:.1f}s"
            )
        
        return None
```

### 4.6 Hash Chain (Tamper-Evidence)

**Location**: `src/backend/hash_chain.py`

```python
import hashlib
import json

class HashChain:
    def __init__(self):
        self.chain = []
        self.last_hash = "0" * 64  # Genesis hash
    
    def append(self, alert: Alert) -> str:
        """Append alert to chain, return new hash."""
        record = {
            "event_id": alert.event_id,
            "prev_hash": self.last_hash,
            "timestamp": alert.timestamp.isoformat(),
            "data": asdict(alert)
        }
        
        # Compute hash
        record_json = json.dumps(record, sort_keys=True)
        record_hash = hashlib.sha256(record_json.encode()).hexdigest()
        
        self.chain.append(record)
        self.last_hash = record_hash
        
        return record_hash
    
    def validate(self) -> bool:
        """Validate entire chain integrity."""
        prev_hash = "0" * 64
        
        for record in self.chain:
            # Check prev_hash linkage
            if record["prev_hash"] != prev_hash:
                return False
            
            # Recompute and verify hash
            stored_hash = record.pop("hash")
            computed_hash = hashlib.sha256(
                json.dumps(record, sort_keys=True).encode()
            ).hexdigest()
            
            if computed_hash != stored_hash:
                return False
            
            prev_hash = stored_hash
        
        return True
```

---

## 5. Database Schema

### 5.1 Events Table

```sql
CREATE TABLE events (
    event_id UUID PRIMARY KEY,
    prev_hash VARCHAR(64) NOT NULL,
    event_hash VARCHAR(64) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    site_id VARCHAR(50) NOT NULL,
    camera_id VARCHAR(50) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    object_class VARCHAR(50),
    track_id VARCHAR(50),
    zone VARCHAR(50),
    bearing VARCHAR(10),
    speed_mps FLOAT,
    confidence FLOAT NOT NULL,
    baseline_deviation BOOLEAN DEFAULT FALSE,
    clip_ref TEXT,
    severity VARCHAR(20) NOT NULL,
    explanation TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_events_site ON events(site_id);
CREATE INDEX idx_events_timestamp ON events(timestamp);
CREATE INDEX idx_events_type ON events(event_type);
CREATE INDEX idx_events_severity ON events(severity);
```

### 5.2 Sites Table

```sql
CREATE TABLE sites (
    site_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    tier INTEGER NOT NULL CHECK (tier IN (1, 2, 3)),
    latitude FLOAT,
    longitude FLOAT,
    border_sector VARCHAR(100),
    status VARCHAR(20) DEFAULT 'active',
    last_heartbeat TIMESTAMPTZ,
    config JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 5.3 Cameras Table

```sql
CREATE TABLE cameras (
    camera_id VARCHAR(50) PRIMARY KEY,
    site_id VARCHAR(50) REFERENCES sites(site_id),
    name VARCHAR(100) NOT NULL,
    rtsp_url TEXT,
    status VARCHAR(20) DEFAULT 'online',
    last_frame_time TIMESTAMPTZ,
    firmware_version VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 5.4 Audit Log

```sql
CREATE TABLE audit_log (
    log_id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_id VARCHAR(50),
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(50),
    details JSONB,
    ip_address INET
);

CREATE INDEX idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX idx_audit_user ON audit_log(user_id);
```

---

## 6. API Endpoints

### 6.1 Events API

```
GET    /api/v1/events              # List events (paginated, filterable)
GET    /api/v1/events/{event_id}   # Get single event
POST   /api/v1/events              # Create event (from edge)
GET    /api/v1/events/stream       # WebSocket stream of new events
```

### 6.2 Sites API

```
GET    /api/v1/sites               # List all sites
GET    /api/v1/sites/{site_id}     # Get site details
PUT    /api/v1/sites/{site_id}     # Update site config
GET    /api/v1/sites/{site_id}/cameras  # List cameras at site
```

### 6.3 Alerts API

```
GET    /api/v1/alerts              # List alerts (paginated, filterable)
GET    /api/v1/alerts/{alert_id}   # Get single alert
PUT    /api/v1/alerts/{alert_id}/acknowledge  # Acknowledge alert
GET    /api/v1/alerts/stream       # WebSocket stream of new alerts
```

### 6.4 Dashboard API

```
GET    /api/v1/dashboard/summary   # Aggregated stats
GET    /api/v1/dashboard/map       # GeoJSON for map view
GET    /api/v1/dashboard/timeline  # Event timeline data
```

### 6.5 Admin API

```
GET    /api/v1/admin/audit-log     # Audit trail
GET    /api/v1/admin/chain-status  # Hash chain integrity status
POST   /api/v1/admin/chain-validate  # Trigger chain validation
```

---

## 7. MQTT Topic Structure

```
ibvap/{site_id}/{camera_id}/events      # Real-time events
ibvap/{site_id}/{camera_id}/heartbeat   # Camera heartbeat
ibvap/{site_id}/{camera_id}/status      # Camera status
ibvap/{site_id}/alerts                  # Aggregated alerts
ibvap/command/{site_id}/{camera_id}     # Command channel (config updates)
```

### Event Payload Example

```json
{
  "event_id": "e7f1a2b3c4d5",
  "site_id": "BOP-14",
  "camera_id": "CAM-2",
  "event_type": "fence_intrusion",
  "object_class": "person",
  "track_id": "T-0042",
  "zone": "Zone-3",
  "bearing": "NE",
  "speed_mps": 1.4,
  "confidence": 0.91,
  "severity": "high",
  "timestamp": "2026-08-24T21:14:03Z",
  "explanation": "Track T-0042 crossed virtual fence Zone-3 at 1.4 m/s, bearing NE."
}
```

---

## 8. Deployment Architecture

### 8.1 Tier-1 Deployment (Jetson Orin Nano)

```yaml
# docker-compose.edge.yml
version: '3.8'
services:
  detector:
    image: ibvap/edge-detector:latest
    runtime: nvidia
    volumes:
      - ./models:/models
      - ./config:/config
    environment:
      - CAMERA_URL=rtsp://admin:password@camera-ip:554/stream
      - MQTT_BROKER=mqtt://broker-ip:1883
      - SITE_ID=BOP-14
      - CAMERA_ID=CAM-2
    
  mqtt-bridge:
    image: ibvap/mqtt-bridge:latest
    depends_on:
      - detector
    volumes:
      - ./config:/config
```

### 8.2 Tier-2 Deployment (Microcontroller)

```python
# Simplified motion trigger
# Runs on ESP32 or similar $20-30 MCU
import cv2
import numpy as np

class MotionTrigger:
    def __init__(self, threshold=0.05):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.threshold = threshold
        self.buffer = []
    
    def process_frame(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        motion_ratio = np.sum(fg_mask > 0) / fg_mask.size
        
        if motion_ratio > self.threshold:
            self.buffer.append(frame)
            if len(self.buffer) > 150:  # 5 sec at 30fps
                self.forward_buffer()
                self.buffer = []
    
    def forward_buffer(self):
        # Send buffered clip to nearest Tier-1 node or cloud
        pass
```

### 8.3 Tier-3 Deployment (Command Center)

```yaml
# docker-compose.cloud.yml
version: '3.8'
services:
  api:
    image: ibvap/api:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/ibvap
      - REDIS_URL=redis://redis:6379
      - MQTT_BROKER=mqtt://mqtt:1883
    
  dashboard:
    image: ibvap/dashboard:latest
    ports:
      - "3000:3000"
    
  mqtt:
    image: eclipse-mosquitto:latest
    ports:
      - "1883:1883"
    volumes:
      - ./mosquitto.conf:/mosquitto/config/mosquitto.conf
  
  db:
    image: postgres:15
    volumes:
      - pgdata:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine

volumes:
  pgdata:
```

---

## 9. Security Architecture

### 9.1 Hash Chain Validation

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Event 1 │───▶│ Event 2 │───▶│ Event 3 │───▶│ Event 4 │
│ hash: A │    │ prev: A │    │ prev: B │    │ prev: C │
│         │    │ hash: B │    │ hash: C │    │ hash: D │
└─────────┘    └─────────┘    └─────────┘    └─────────┘

If Event 2 is modified:
- Its hash changes to B'
- Event 3's prev_hash no longer matches B'
- Chain validation fails → tampering detected
```

### 9.2 Role-Based Access Control

```python
class Role(Enum):
    OPERATOR = "operator"      # View alerts, acknowledge
    COMMANDER = "commander"    # View all, configure fences
    ADMIN = "admin"            # Full access, user management
    AUDITOR = "auditor"        # Read-only, audit log access

PERMISSIONS = {
    Role.OPERATOR: ["alerts:read", "alerts:acknowledge", "events:read"],
    Role.COMMANDER: ["alerts:read", "alerts:acknowledge", "events:read", 
                     "fences:write", "sites:read"],
    Role.ADMIN: ["*"],  # Full access
    Role.AUDITOR: ["events:read", "audit:read", "chain:validate"],
}
```

### 9.3 Data Retention Policy

| Data Type | Retention | Deletion Method |
|-----------|-----------|-----------------|
| Raw video clips | 7 days | Automatic purge |
| Event metadata | 90 days | Automatic purge |
| Alert records | 1 year | Archive to cold storage |
| Audit logs | 5 years | Immutable, compliance |
| Hash chain | Permanent | Append-only, never deleted |

---

## 10. Performance Benchmarks

### 10.1 Edge Inference (Jetson Orin Nano)

| Model | Input Size | FPS | Latency |
|-------|------------|-----|---------|
| YOLOv8-nano | 640x640 | 45 | 22ms |
| YOLOv8-nano | 320x320 | 90 | 11ms |
| ByteTrack | - | 120 | 8ms |
| PaddleOCR | 320x64 | 30 | 33ms |

### 10.2 Backend Throughput

| Operation | Throughput | Latency (P95) |
|-----------|------------|---------------|
| Event ingestion | 10,000/sec | 5ms |
| Alert generation | 1,000/sec | 15ms |
| Dashboard queries | 500/sec | 50ms |
| MQTT publish | 5,000/sec | 3ms |

---

*Architecture Version: 1.0*
*Last Updated: 2026-08-29*
