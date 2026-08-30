"""Virtual fence — containment, entry/exit edges, severity and cooldown.

The fence fires on the *transition* into a zone, not on presence, so most
of these tests are about edges: an object sitting inside must not re-alert
every frame, and one that leaves and returns must alert again.
"""
import time

from src.edge.fence import VirtualFence

SQUARE = [(100, 100), (300, 100), (300, 300), (100, 300)]


def make_fence(cooldown=0.0):
    fence = VirtualFence(cooldown_seconds=cooldown)
    fence.add_zone("Test Zone", SQUARE, severity="high", description="test")
    return fence


def test_point_inside_polygon_is_contained():
    fence = make_fence()
    assert fence.zones["Test Zone"].contains((200, 200)) is True


def test_point_outside_polygon_is_not_contained():
    fence = make_fence()
    assert fence.zones["Test Zone"].contains((50, 50)) is False


def test_entering_a_zone_raises_one_event():
    fence = make_fence()
    events = fence.check_intrusion(track_id=1, center=(200, 200), object_class="person")
    assert len(events) == 1
    assert events[0].zone_name == "Test Zone"
    assert events[0].track_id == 1


def test_staying_inside_does_not_re_alert():
    """Presence is not an event — only the crossing is."""
    fence = make_fence()
    assert len(fence.check_intrusion(1, (200, 200))) == 1
    assert fence.check_intrusion(1, (210, 210)) == []
    assert fence.check_intrusion(1, (220, 220)) == []


def test_staying_outside_never_alerts():
    fence = make_fence()
    for x in range(0, 90, 10):
        assert fence.check_intrusion(1, (x, 50)) == []


def test_leaving_and_re_entering_alerts_again():
    fence = make_fence(cooldown=0.0)
    assert len(fence.check_intrusion(1, (200, 200))) == 1
    assert fence.check_intrusion(1, (50, 50)) == []       # exit
    assert len(fence.check_intrusion(1, (200, 200))) == 1  # re-entry


def test_cooldown_suppresses_a_rapid_second_alert():
    fence = make_fence(cooldown=60.0)
    assert len(fence.check_intrusion(1, (200, 200))) == 1
    fence.check_intrusion(1, (50, 50))                     # exit
    assert fence.check_intrusion(1, (200, 200)) == []      # within cooldown


def test_cooldown_expires():
    fence = make_fence(cooldown=0.05)
    assert len(fence.check_intrusion(1, (200, 200))) == 1
    fence.check_intrusion(1, (50, 50))
    time.sleep(0.06)
    assert len(fence.check_intrusion(1, (200, 200))) == 1


def test_event_carries_the_explanation_fields():
    fence = make_fence()
    event = fence.check_intrusion(
        track_id=7, center=(200, 200), object_class="car",
        speed=3.5, bearing="N", confidence=0.88,
    )[0]
    assert event.object_class == "car"
    assert event.bearing == "N"
    assert event.speed_mps == 3.5
    assert "car" in event.explanation and "Test Zone" in event.explanation

    as_dict = event.to_dict()
    for key in ("zone_name", "track_id", "object_class", "point", "explanation"):
        assert key in as_dict


def test_inactive_zone_does_not_alert():
    fence = make_fence()
    fence.toggle_zone("Test Zone")
    assert fence.check_intrusion(1, (200, 200)) == []


def test_removed_zone_does_not_alert():
    fence = make_fence()
    fence.remove_zone("Test Zone")
    assert fence.check_intrusion(1, (200, 200)) == []
    assert len(fence.zones) == 0


def test_two_tracks_alert_independently():
    fence = make_fence(cooldown=0.0)
    assert len(fence.check_intrusion(1, (200, 200))) == 1
    assert len(fence.check_intrusion(2, (210, 210))) == 1


def test_default_zone_creates_one_zone():
    fence = VirtualFence()
    fence.add_default_zone(640, 480)
    assert len(fence.zones) == 1


def test_demo_preset_creates_zones_with_differing_severity():
    """The preset exists so a demo can show severity varying by zone."""
    fence = VirtualFence()
    fence.add_demo_preset(640, 480)
    assert len(fence.zones) == 3
    severities = {zone.severity for zone in fence.zones.values()}
    assert len(severities) > 1, f"all preset zones share one severity: {severities}"


def test_demo_preset_zones_are_disjoint_enough_to_distinguish():
    """A point in the vehicle lane must not also be in the pedestrian zone."""
    fence = VirtualFence()
    fence.add_demo_preset(640, 480)
    lane_point = (320, 460)   # bottom strip
    hits = [name for name, zone in fence.zones.items() if zone.contains(lane_point)]
    assert hits == ["Vehicle Lane"]
