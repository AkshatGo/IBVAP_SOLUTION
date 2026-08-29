# IBVAP — API Schemas & Data Models

---

## 1. Core Data Models

### 1.1 Event

```python
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum

class EventType(str, Enum):
    FENCE_INTRUSION = "fence_intrusion"
    ANPR_MATCH = "anpr_match"
    ANPR_UNKNOWN = "anpr_unknown"
    HUMAN_DETECTION = "human_detection"
    VEHICLE_DETECTION = "vehicle_detection"
    SIGNAL_LOSS = "signal_loss"
    BEHAVIORAL_DEVIATION = "behavioral_deviation"
    ZONE_ENTRY = "zone_entry"
    ZONE_EXIT = "zone_exit"

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Event(BaseModel):
    event_id: str = Field(..., description="Unique event identifier")
    prev_hash: str = Field(..., description="Hash of previous event in chain")
    event_hash: str = Field(..., description="SHA-256 hash of this event")
    timestamp: datetime = Field(..., description="Event timestamp (UTC)")
    site_id: str = Field(..., description="Border post identifier")
    camera_id: str = Field(..., description="Camera identifier")
    event_type: EventType = Field(..., description="Type of event detected")
    object_class: Optional[str] = Field(None, description="Detected object class")
    track_id: Optional[str] = Field(None, description="Tracking ID across frames")
    zone: Optional[str] = Field(None, description="Geofence zone name")
    bearing: Optional[str] = Field(None, description="Direction of movement")
    speed_mps: Optional[float] = Field(None, description="Speed in meters/second")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    baseline_deviation: bool = Field(False, description="Deviation from behavioral baseline")
    clip_ref: Optional[str] = Field(None, description="Reference to video clip")
    severity: Severity = Field(..., description="Alert severity level")
    explanation: str = Field(..., description="Human-readable explanation of alert")
    metadata: Optional[dict] = Field(None, description="Additional event data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "e7f1a2b3c4d5e6f7",
                "prev_hash": "a92c8b7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f0a9b",
                "event_hash": "b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2",
                "timestamp": "2026-08-24T21:14:03Z",
                "site_id": "BOP-14",
                "camera_id": "CAM-2",
                "event_type": "fence_intrusion",
                "object_class": "person",
                "track_id": "T-0042",
                "zone": "Zone-3",
                "bearing": "NE",
                "speed_mps": 1.4,
                "confidence": 0.91,
                "baseline_deviation": True,
                "clip_ref": "s3://ibvap-clips/BOP-14/CAM-2/e7f1a2b3.mp4",
                "severity": "high",
                "explanation": "Track T-0042 crossed virtual fence Zone-3 at 1.4 m/s, bearing NE. No scheduled patrol active in this zone at this time."
            }
        }
```

### 1.2 Site

```python
class SiteTier(int, Enum):
    TIER_1 = 1  # High-priority BOP with edge compute
    TIER_2 = 2  # Remote road camera, minimal compute
    TIER_3 = 3  # Command center

class SiteStatus(str, Enum):
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"

class Site(BaseModel):
    site_id: str = Field(..., description="Unique site identifier")
    name: str = Field(..., description="Human-readable site name")
    tier: SiteTier = Field(..., description="Deployment tier")
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    border_sector: Optional[str] = Field(None, description="Border sector name")
    status: SiteStatus = Field(SiteStatus.ACTIVE)
    last_heartbeat: Optional[datetime] = None
    config: Optional[dict] = Field(None, description="Site-specific configuration")
    
    class Config:
        json_schema_extra = {
            "example": {
                "site_id": "BOP-14",
                "name": "Border Out Post 14 - Pangong",
                "tier": 1,
                "latitude": 33.75,
                "longitude": 78.68,
                "border_sector": "Ladakh Sector",
                "status": "active"
            }
        }
```

### 1.3 Camera

```python
class CameraStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    SIGNAL_LOST = "signal_lost"

class Camera(BaseModel):
    camera_id: str = Field(..., description="Unique camera identifier")
    site_id: str = Field(..., description="Parent site identifier")
    name: str = Field(..., description="Human-readable camera name")
    rtsp_url: Optional[str] = Field(None, description="RTSP stream URL")
    status: CameraStatus = Field(CameraStatus.ONLINE)
    last_frame_time: Optional[datetime] = None
    firmware_version: Optional[str] = None
    resolution: Optional[str] = Field(None, description="e.g., 1920x1080")
    fps: Optional[int] = Field(None, description="Frames per second")
    
    class Config:
        json_schema_extra = {
            "example": {
                "camera_id": "CAM-2",
                "site_id": "BOP-14",
                "name": "North Gate Camera",
                "rtsp_url": "rtsp://admin:pass@192.168.1.100:554/stream",
                "status": "online",
                "resolution": "1920x1080",
                "fps": 30
            }
        }
```

### 1.4 Virtual Fence

```python
class Point(BaseModel):
    x: float = Field(..., description="X coordinate (pixels or lat)")
    y: float = Field(..., description="Y coordinate (pixels or lon)")

class VirtualFence(BaseModel):
    fence_id: str = Field(..., description="Unique fence identifier")
    site_id: str = Field(..., description="Site this fence belongs to")
    camera_id: str = Field(..., description="Camera this fence is defined on")
    name: str = Field(..., description="Human-readable fence name")
    polygon: List[Point] = Field(..., min_length=3, description="Polygon vertices")
    zone: str = Field(..., description="Zone identifier")
    severity_override: Optional[Severity] = Field(None, description="Override alert severity")
    enabled: bool = Field(True, description="Whether fence is active")
    
    class Config:
        json_schema_extra = {
            "example": {
                "fence_id": "FENCE-001",
                "site_id": "BOP-14",
                "camera_id": "CAM-2",
                "name": "North Perimeter",
                "polygon": [
                    {"x": 100, "y": 100},
                    {"x": 500, "y": 100},
                    {"x": 500, "y": 400},
                    {"x": 100, "y": 400}
                ],
                "zone": "Zone-3",
                "enabled": True
            }
        }
```

### 1.5 Alert (Dashboard View)

```python
class Alert(BaseModel):
    alert_id: str = Field(..., description="Unique alert identifier")
    event: Event = Field(..., description="Underlying event")
    acknowledged: bool = Field(False)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "alert_id": "ALT-20260824-001",
                "event": {"event_id": "e7f1a2b3c4d5e6f7", "...": "..."},
                "acknowledged": False,
                "created_at": "2026-08-24T21:14:03Z"
            }
        }
```

### 1.6 User & Authentication

```python
class UserRole(str, Enum):
    OPERATOR = "operator"
    COMMANDER = "commander"
    ADMIN = "admin"
    AUDITOR = "auditor"

class User(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., description="User email")
    role: UserRole = Field(..., description="User role")
    sites: List[str] = Field(default_factory=list, description="Assigned site IDs")
    enabled: bool = Field(True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token TTL in seconds")
    user: User
```

---

## 2. API Request/Response Schemas

### 2.1 Events API

#### List Events

```python
class EventListRequest(BaseModel):
    site_id: Optional[str] = None
    camera_id: Optional[str] = None
    event_type: Optional[EventType] = None
    severity: Optional[Severity] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = Field(50, ge=1, le=200)
    offset: int = Field(0, ge=0)

class EventListResponse(BaseModel):
    events: List[Event]
    total: int
    limit: int
    offset: int
    has_more: bool
```

#### Create Event (from Edge)

```python
class CreateEventRequest(BaseModel):
    site_id: str
    camera_id: str
    event_type: EventType
    object_class: Optional[str] = None
    track_id: Optional[str] = None
    zone: Optional[str] = None
    bearing: Optional[str] = None
    speed_mps: Optional[float] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    clip_data: Optional[bytes] = Field(None, description="Base64-encoded clip")
    metadata: Optional[dict] = None

class CreateEventResponse(BaseModel):
    event_id: str
    severity: Severity
    explanation: str
    created_at: datetime
```

### 2.2 Sites API

#### List Sites

```python
class SiteListResponse(BaseModel):
    sites: List[Site]
    total: int
```

#### Update Site Config

```python
class UpdateSiteRequest(BaseModel):
    name: Optional[str] = None
    tier: Optional[SiteTier] = None
    status: Optional[SiteStatus] = None
    config: Optional[dict] = None

class UpdateSiteResponse(BaseModel):
    site: Site
    updated_at: datetime
```

### 2.3 Alerts API

#### List Alerts

```python
class AlertListRequest(BaseModel):
    site_id: Optional[str] = None
    severity: Optional[Severity] = None
    acknowledged: Optional[bool] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = Field(50, ge=1, le=200)
    offset: int = Field(0, ge=0)

class AlertListResponse(BaseModel):
    alerts: List[Alert]
    total: int
    unread_count: int
    limit: int
    offset: int
```

#### Acknowledge Alert

```python
class AcknowledgeAlertRequest(BaseModel):
    notes: Optional[str] = None

class AcknowledgeAlertResponse(BaseModel):
    alert: Alert
    acknowledged_at: datetime
```

### 2.4 Dashboard API

#### Summary

```python
class DashboardSummaryResponse(BaseModel):
    total_sites: int
    active_sites: int
    total_cameras: int
    online_cameras: int
    alerts_today: int
    alerts_unacknowledged: int
    events_today: int
    average_confidence: float
    top_event_types: List[dict]
    severity_distribution: dict

class MapDataResponse(BaseModel):
    type: str = "FeatureCollection"
    features: List[dict]  # GeoJSON features for each site/camera
```

### 2.5 Admin API

#### Hash Chain Status

```python
class ChainStatusResponse(BaseModel):
    total_events: int
    chain_length: int
    last_hash: str
    is_valid: bool
    last_validated: Optional[datetime]
    tampered_events: List[str] = Field(default_factory=list)

class ChainValidateRequest(BaseModel):
    start_event_id: Optional[str] = None
    end_event_id: Optional[str] = None

class ChainValidateResponse(BaseModel):
    valid: bool
    events_checked: int
    first_invalid_event: Optional[str]
    validation_time_ms: float
```

---

## 3. MQTT Message Schemas

### 3.1 Event Message

```json
{
  "topic": "ibvap/BOP-14/CAM-2/events",
  "payload": {
    "event_id": "e7f1a2b3c4d5e6f7",
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
  },
  "qos": 1,
  "retain": false
}
```

### 3.2 Heartbeat Message

```json
{
  "topic": "ibvap/BOP-14/CAM-2/heartbeat",
  "payload": {
    "camera_id": "CAM-2",
    "site_id": "BOP-14",
    "timestamp": "2026-08-24T21:14:03Z",
    "fps": 30,
    "cpu_usage": 45.2,
    "gpu_usage": 78.5,
    "temperature": 62.0,
    "disk_usage_percent": 34.0,
    "uptime_seconds": 86400
  },
  "qos": 0,
  "retain": true
}
```

### 3.3 Signal Loss Alert

```json
{
  "topic": "ibvap/BOP-14/alerts",
  "payload": {
    "event_id": "sl-001",
    "site_id": "BOP-14",
    "camera_id": "CAM-2",
    "event_type": "signal_loss",
    "severity": "high",
    "timestamp": "2026-08-24T21:14:03Z",
    "explanation": "Camera CAM-2 signal lost for 5.2 seconds. Possible jamming or hardware failure."
  },
  "qos": 2,
  "retain": false
}
```

### 3.4 Command Message

```json
{
  "topic": "ibvap/command/BOP-14/CAM-2",
  "payload": {
    "command": "update_config",
    "parameters": {
      "detection_threshold": 0.6,
      "tracking_max_age": 30,
      "fence_ids": ["FENCE-001", "FENCE-002"]
    },
    "request_id": "req-abc123",
    "timestamp": "2026-08-24T21:14:03Z"
  },
  "qos": 1,
  "retain": false
}
```

---

## 4. WebSocket Events

### 4.1 Real-time Alert Stream

```javascript
// Client connects to WebSocket
const ws = new WebSocket('ws://api.ibvap.local:8000/api/v1/alerts/stream');

// Server sends alert events
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    // data.type: "new_alert" | "alert_acknowledged" | "alert_updated"
    // data.alert: Alert object
};

// Filter by site
ws.send(JSON.stringify({
    action: "subscribe",
    filters: {
        site_ids: ["BOP-14", "BOP-15"],
        min_severity: "medium"
    }
}));
```

### 4.2 Event Stream

```javascript
// Client connects to WebSocket
const ws = new WebSocket('ws://api.ibvap.local:8000/api/v1/events/stream');

// Server sends event data
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    // data.type: "new_event" | "event_updated"
    // data.event: Event object
};
```

---

## 5. Error Responses

### 5.1 Standard Error Format

```python
class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(None, description="Additional error details")
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### 5.2 Common Error Codes

```python
class ErrorCode(str, Enum):
    # Authentication
    AUTH_REQUIRED = "auth_required"
    AUTH_INVALID_TOKEN = "auth_invalid_token"
    AUTH_INSUFFICIENT_PERMISSIONS = "auth_insufficient_permissions"
    
    # Validation
    VALIDATION_ERROR = "validation_error"
    INVALID_EVENT_TYPE = "invalid_event_type"
    INVALID_SEVERITY = "invalid_severity"
    
    # Resources
    EVENT_NOT_FOUND = "event_not_found"
    SITE_NOT_FOUND = "site_not_found"
    CAMERA_NOT_FOUND = "camera_not_found"
    ALERT_NOT_FOUND = "alert_not_found"
    
    # System
    INTERNAL_ERROR = "internal_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    
    # Chain
    CHAIN_VALIDATION_FAILED = "chain_validation_failed"
    CHAIN_TAMPERED = "chain_tampered"
```

### 5.3 Error Examples

```json
{
    "error": "validation_error",
    "message": "Invalid event type: 'unknown_type'",
    "details": {
        "field": "event_type",
        "invalid_value": "unknown_type",
        "valid_values": ["fence_intrusion", "anpr_match", "human_detection", "vehicle_detection", "signal_loss"]
    },
    "request_id": "req-abc123",
    "timestamp": "2026-08-24T21:14:03Z"
}
```

---

## 6. Database Models (SQLAlchemy)

```python
from sqlalchemy import Column, String, Float, Boolean, DateTime, Text, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid

class EventModel(Base):
    __tablename__ = "events"
    
    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prev_hash = Column(String(64), nullable=False)
    event_hash = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    site_id = Column(String(50), ForeignKey("sites.site_id"), nullable=False, index=True)
    camera_id = Column(String(50), ForeignKey("cameras.camera_id"), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    object_class = Column(String(50))
    track_id = Column(String(50))
    zone = Column(String(50))
    bearing = Column(String(10))
    speed_mps = Column(Float)
    confidence = Column(Float, nullable=False)
    baseline_deviation = Column(Boolean, default=False)
    clip_ref = Column(Text)
    severity = Column(String(20), nullable=False, index=True)
    explanation = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    site = relationship("SiteModel", back_populates="events")
    camera = relationship("CameraModel", back_populates="events")
    
    __table_args__ = (
        Index("idx_events_site_timestamp", "site_id", "timestamp"),
        Index("idx_events_type_severity", "event_type", "severity"),
    )

class SiteModel(Base):
    __tablename__ = "sites"
    
    site_id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    tier = Column(Integer, nullable=False)
    latitude = Column(Float)
    longitude = Column(Float)
    border_sector = Column(String(100))
    status = Column(String(20), default="active")
    last_heartbeat = Column(DateTime(timezone=True))
    config = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    events = relationship("EventModel", back_populates="site")
    cameras = relationship("CameraModel", back_populates="site")

class CameraModel(Base):
    __tablename__ = "cameras"
    
    camera_id = Column(String(50), primary_key=True)
    site_id = Column(String(50), ForeignKey("sites.site_id"), nullable=False)
    name = Column(String(100), nullable=False)
    rtsp_url = Column(Text)
    status = Column(String(20), default="online")
    last_frame_time = Column(DateTime(timezone=True))
    firmware_version = Column(String(50))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    site = relationship("SiteModel", back_populates="cameras")
    events = relationship("EventModel", back_populates="camera")

class AuditLogModel(Base):
    __tablename__ = "audit_log"
    
    log_id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow, index=True)
    user_id = Column(String(50), index=True)
    action = Column(String(50), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(String(50))
    details = Column(JSONB)
    ip_address = Column(String(45))  # IPv4 or IPv6
```

---

*Schema Version: 1.0*
*Last Updated: 2026-08-29*
