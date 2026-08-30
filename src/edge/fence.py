"""
Virtual Fence — Polygon-based intrusion detection.
Users draw fence zones on camera feeds; system alerts when tracked objects cross.
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import time


@dataclass
class FenceZone:
    """A named virtual fence zone."""
    name: str
    polygon: np.ndarray  # Nx2 array of (x,y) points
    severity: str = "high"
    is_active: bool = True
    description: str = ""

    def contains(self, point: Tuple[int, int]) -> bool:
        """Check if a point is inside this fence zone."""
        result = cv2.pointPolygonTest(
            self.polygon.astype(np.float32),
            (float(point[0]), float(point[1])),
            False
        )
        return result >= 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "polygon": self.polygon.tolist(),
            "severity": self.severity,
            "is_active": self.is_active,
            "description": self.description,
        }


@dataclass
class IntrusionEvent:
    """A fence intrusion event."""
    zone_name: str
    track_id: int
    object_class: str
    point: Tuple[int, int]
    timestamp: float
    speed_mps: float = 0.0
    bearing: str = "unknown"
    confidence: float = 0.0
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "zone_name": self.zone_name,
            "track_id": self.track_id,
            "object_class": self.object_class,
            "point": self.point,
            "timestamp": self.timestamp,
            "speed_mps": round(self.speed_mps, 2),
            "bearing": self.bearing,
            "confidence": round(self.confidence, 3),
            "explanation": self.explanation,
        }


class VirtualFence:
    """
    Virtual fence system with multiple zones and cooldown.
    
    Features:
    - Multiple named zones with different severity levels
    - Cooldown to prevent alert spam
    - Entry/exit tracking
    - Speed and bearing calculation
    """

    def __init__(self, cooldown_seconds: float = 5.0):
        self.zones: Dict[str, FenceZone] = {}
        self.cooldown_seconds = cooldown_seconds
        self._last_alert: Dict[str, float] = {}  # zone_name -> timestamp
        self._inside: Dict[str, set] = {}  # zone_name -> set of track_ids inside

    def add_zone(self, name: str, polygon: List[Tuple[int, int]],
                 severity: str = "high", description: str = ""):
        """Add a fence zone."""
        pts = np.array(polygon, dtype=np.int32)
        self.zones[name] = FenceZone(
            name=name, polygon=pts, severity=severity, description=description
        )
        self._inside[name] = set()

    def add_default_zone(self, frame_width: int = 640, frame_height: int = 480):
        """Add a default single perimeter fence."""
        margin_x = int(frame_width * 0.28)
        margin_y = int(frame_height * 0.25)
        polygon = [
            (margin_x, margin_y),
            (frame_width - margin_x, margin_y),
            (frame_width - margin_x, frame_height - margin_y),
            (margin_x, frame_height - margin_y),
        ]
        self.add_zone("Zone-1", polygon, severity="high",
                       description="Primary perimeter zone")

    def add_demo_preset(self, frame_width: int = 640, frame_height: int = 480):
        """Add a multi-zone demo preset: pedestrian zone + vehicle lane + critical zone.

        This gives a richer demo where different zones have different
        severity levels and you can show multi-zone behavior live.
        """
        # Pedestrian zone — upper-center area
        ped_margin_x = int(frame_width * 0.30)
        ped_margin_y = int(frame_height * 0.20)
        pedestrian_zone = [
            (ped_margin_x, ped_margin_y),
            (frame_width - ped_margin_x, ped_margin_y),
            (frame_width - ped_margin_x, int(frame_height * 0.55)),
            (ped_margin_x, int(frame_height * 0.55)),
        ]
        self.add_zone("Pedestrian Zone", pedestrian_zone, severity="high",
                       description="Restricted pedestrian area — high alert on intrusion")

        # Vehicle lane — bottom strip
        vehicle_zone = [
            (0, int(frame_height * 0.70)),
            (frame_width, int(frame_height * 0.70)),
            (frame_width, frame_height - 10),
            (0, frame_height - 10),
        ]
        self.add_zone("Vehicle Lane", vehicle_zone, severity="medium",
                       description="Vehicle passage lane — medium alert")

        # Critical zone — small inner area
        crit_cx, crit_cy = frame_width // 2, int(frame_height * 0.38)
        crit_half = 40
        critical_zone = [
            (crit_cx - crit_half, crit_cy - crit_half),
            (crit_cx + crit_half, crit_cy - crit_half),
            (crit_cx + crit_half, crit_cy + crit_half),
            (crit_cx - crit_half, crit_cy + crit_half),
        ]
        self.add_zone("Critical Zone", critical_zone, severity="critical",
                       description="Inner critical area — any intrusion is critical")

    def check_intrusion(self, track_id: int, center: Tuple[int, int],
                        object_class: str = "unknown",
                        speed: float = 0.0, bearing: str = "unknown",
                        confidence: float = 0.0) -> List[IntrusionEvent]:
        """
        Check if a tracked object has entered any fence zone.
        Returns list of new intrusion events (with cooldown applied).
        """
        events = []
        now = time.time()

        for zone_name, zone in self.zones.items():
            if not zone.is_active:
                continue

            is_inside = zone.contains(center)
            was_inside = track_id in self._inside.get(zone_name, set())

            if is_inside and not was_inside:
                # New entry!
                last_time = self._last_alert.get(zone_name, 0)
                if now - last_time >= self.cooldown_seconds:
                    explanation = (
                        f"Track T-{track_id:04d} ({object_class}) crossed "
                        f"virtual fence '{zone.name}' at {speed:.1f} m/s, "
                        f"bearing {bearing}."
                    )
                    event = IntrusionEvent(
                        zone_name=zone.name,
                        track_id=track_id,
                        object_class=object_class,
                        point=center,
                        timestamp=now,
                        speed_mps=speed,
                        bearing=bearing,
                        confidence=confidence,
                        explanation=explanation,
                    )
                    events.append(event)
                    self._last_alert[zone_name] = now

            # Update inside state
            if is_inside:
                self._inside[zone_name].add(track_id)
            else:
                self._inside[zone_name].discard(track_id)

        return events

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw all fence zones on the frame."""
        vis = frame.copy()
        for zone_name, zone in self.zones.items():
            color = (0, 255, 0) if zone.is_active else (128, 128, 128)
            cv2.polylines(vis, [zone.polygon], True, color, 2)

            # Label
            centroid = zone.polygon.mean(axis=0).astype(int)
            cv2.putText(vis, zone.name, (centroid[0] - 30, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Show objects inside
            inside_ids = self._inside.get(zone_name, set())
            if inside_ids:
                cv2.putText(vis, f"{len(inside_ids)} inside",
                            (centroid[0] - 30, centroid[1] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        return vis

    def remove_zone(self, name: str):
        """Remove a fence zone."""
        self.zones.pop(name, None)
        self._inside.pop(name, None)
        self._last_alert.pop(name, None)

    def toggle_zone(self, name: str):
        """Toggle a zone active/inactive."""
        if name in self.zones:
            self.zones[name].is_active = not self.zones[name].is_active

    def get_all_zones(self) -> List[dict]:
        """Return all zones as dicts."""
        return [z.to_dict() for z in self.zones.values()]
