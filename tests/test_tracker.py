"""Object tracker — ID stability, association, and aging.

The headline test here is `test_id_is_stable_for_a_moving_object`. The
tracker's previous fallback assigned a fresh ID every frame, which silently
broke fence entry/exit, speed estimation and ANPR consensus — all of which
depend on a track_id meaning the same object over time. That failure was
invisible without a test, so it gets one.
"""
from src.edge.tracker import ObjectTracker, _iou
from conftest import make_detection

SHAPE = (480, 640)


def test_iou_of_identical_boxes_is_one():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_of_disjoint_boxes_is_zero():
    assert _iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_iou_of_half_overlap():
    # Two 10x10 boxes overlapping in a 5x10 strip: 50 / 150.
    assert _iou((0, 0, 10, 10), (5, 0, 15, 10)) == 50 / 150


def test_first_detection_gets_an_id():
    tracker = ObjectTracker()
    tracked = tracker.update([make_detection((100, 100, 150, 200))], SHAPE)
    assert len(tracked) == 1
    assert tracked[0].track_id > 0


def test_id_is_stable_for_a_moving_object():
    """A single object drifting across frames must keep one ID."""
    tracker = ObjectTracker()
    ids = []
    for step in range(10):
        # 4px per frame keeps IoU well above the 0.8 match threshold.
        bbox = (100 + step * 4, 100, 150 + step * 4, 200)
        tracked = tracker.update([make_detection(bbox)], SHAPE)
        ids.append(tracked[0].track_id)

    assert len(set(ids)) == 1, f"track ID churned across frames: {ids}"


def test_two_objects_keep_separate_ids():
    tracker = ObjectTracker()
    first = make_detection((100, 100, 150, 200))
    second = make_detection((400, 100, 450, 200))
    tracked = tracker.update([first, second], SHAPE)
    assert len({d.track_id for d in tracked}) == 2

    # Nudge both; each should retain its own ID.
    ids_before = {(d.bbox[0], d.track_id) for d in tracked}
    tracked = tracker.update(
        [make_detection((104, 100, 154, 200)), make_detection((404, 100, 454, 200))],
        SHAPE,
    )
    assert len({d.track_id for d in tracked}) == 2
    assert {t for _, t in ids_before} == {d.track_id for d in tracked}


def test_a_teleporting_object_gets_a_new_id():
    """No IoU overlap means no association — that is correct, not a bug."""
    tracker = ObjectTracker()
    first = tracker.update([make_detection((100, 100, 150, 200))], SHAPE)[0]
    second = tracker.update([make_detection((500, 300, 550, 400))], SHAPE)[0]
    assert first.track_id != second.track_id


def test_different_classes_never_associate():
    tracker = ObjectTracker()
    person = tracker.update(
        [make_detection((100, 100, 150, 200), class_id=0, class_name="person")], SHAPE
    )[0]
    car = tracker.update(
        [make_detection((100, 100, 150, 200), class_id=2, class_name="car")], SHAPE
    )[0]
    assert person.track_id != car.track_id


def test_low_confidence_detections_are_dropped():
    tracker = ObjectTracker(track_thresh=0.5)
    tracked = tracker.update([make_detection((100, 100, 150, 200), confidence=0.2)], SHAPE)
    assert tracked == []


def test_track_ages_out_after_buffer_expires():
    tracker = ObjectTracker(track_buffer=3)
    tracker.update([make_detection((100, 100, 150, 200))], SHAPE)
    assert tracker.active_track_count() == 1

    for _ in range(5):  # more empty frames than the buffer allows
        tracker.update([], SHAPE)
    assert tracker.active_track_count() == 0


def test_track_survives_a_brief_gap():
    tracker = ObjectTracker(track_buffer=5)
    first = tracker.update([make_detection((100, 100, 150, 200))], SHAPE)[0]
    tracker.update([], SHAPE)  # one dropped frame
    again = tracker.update([make_detection((100, 100, 150, 200))], SHAPE)[0]
    assert again.track_id == first.track_id


def test_confirmation_requires_min_hits():
    tracker = ObjectTracker(min_hits=3)
    tid = tracker.update([make_detection((100, 100, 150, 200))], SHAPE)[0].track_id
    assert tracker.is_confirmed(tid) is False

    for step in range(1, 4):
        tracker.update([make_detection((100 + step, 100, 150 + step, 200))], SHAPE)
    assert tracker.is_confirmed(tid) is True


def test_trajectory_accumulates_centers():
    tracker = ObjectTracker()
    tid = None
    for step in range(4):
        bbox = (100 + step * 4, 100, 150 + step * 4, 200)
        tid = tracker.update([make_detection(bbox)], SHAPE)[0].track_id
    assert len(tracker.get_trajectory(tid)) == 4


def test_speed_is_zero_for_a_stationary_object():
    tracker = ObjectTracker()
    tid = None
    for _ in range(5):
        tid = tracker.update([make_detection((100, 100, 150, 200))], SHAPE)[0].track_id
    assert tracker.get_speed(tid) == 0.0


def test_speed_is_positive_for_a_moving_object():
    tracker = ObjectTracker()
    tid = None
    for step in range(5):
        bbox = (100 + step * 4, 100, 150 + step * 4, 200)
        tid = tracker.update([make_detection(bbox)], SHAPE)[0].track_id
    assert tracker.get_speed(tid) > 0


def test_bearing_follows_direction_of_travel():
    tracker = ObjectTracker()
    tid = None
    for step in range(3):
        bbox = (100 + step * 4, 100, 150 + step * 4, 200)
        tid = tracker.update([make_detection(bbox)], SHAPE)[0].track_id
    assert tracker.get_bearing(tid) == "E"


def test_reset_clears_all_state():
    tracker = ObjectTracker()
    tracker.update([make_detection((100, 100, 150, 200))], SHAPE)
    tracker.reset()
    assert tracker.active_track_count() == 0
    assert tracker.update([make_detection((100, 100, 150, 200))], SHAPE)[0].track_id == 1
