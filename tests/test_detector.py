"""Detector — class taxonomy adoption.

The bug these guard against is silent and severe: EdgeDetector shipped a
hardcoded COCO index map, and the IDD fine-tuned model uses a different
one. Under COCO, index 6 is "train" and truck is 7; under the fine-tuned
taxonomy index 6 is "autorickshaw". Running the fine-tuned weights against
the COCO map therefore reported every auto-rickshaw as some other class —
mislabelling the exact detection the fine-tune exists to provide, while
producing output that looks entirely plausible.

No real weights are loaded here: the YOLO constructor is stubbed, so the
tests pin the mapping logic without downloading or reading a checkpoint.
"""
import pytest

from src.edge import detector as detector_module
from src.edge.detector import EdgeDetector

# The fine-tuned taxonomy, as written by scripts/idd_remap.py.
IDD_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
             4: "bus", 5: "truck", 6: "autorickshaw"}

# Enough of COCO to be recognisable, padded to 80 so the stock check trips.
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
              5: "bus", 7: "truck"}
COCO_NAMES.update({i: f"coco_{i}" for i in range(80) if i not in COCO_NAMES})


class FakeYOLO:
    """Stands in for ultralytics.YOLO — carries a `names` map and nothing else."""

    def __init__(self, names):
        self.names = names

    def __call__(self, *args, **kwargs):
        return []


@pytest.fixture
def stub_yolo(monkeypatch):
    """Patch the YOLO constructor; the test supplies the taxonomy."""
    def install(names):
        monkeypatch.setattr(detector_module, "YOLO", lambda path: FakeYOLO(names))
    return install


def test_fine_tuned_taxonomy_is_adopted(stub_yolo):
    stub_yolo(IDD_NAMES)
    d = EdgeDetector(model_path="models/weights/detection.pt").load()
    assert d.class_names == IDD_NAMES


def test_autorickshaw_is_not_reported_as_another_class(stub_yolo):
    """The regression itself: index 6 must resolve to autorickshaw."""
    stub_yolo(IDD_NAMES)
    d = EdgeDetector(model_path="models/weights/detection.pt").load()
    assert d.class_names[6] == "autorickshaw"
    assert d.class_names[5] == "truck"


def test_autorickshaw_is_targeted_not_filtered_out(stub_yolo):
    """Adopting names is useless if the class is then dropped by the filter."""
    stub_yolo(IDD_NAMES)
    d = EdgeDetector(model_path="models/weights/detection.pt").load()
    assert 6 in d.target_classes


def test_all_seven_classes_are_targeted(stub_yolo):
    stub_yolo(IDD_NAMES)
    d = EdgeDetector(model_path="models/weights/detection.pt").load()
    assert sorted(d.target_classes) == [0, 1, 2, 3, 4, 5, 6]


def test_stock_coco_indices_are_left_alone(stub_yolo):
    """Stock weights must keep the configured COCO indices, not be remapped."""
    stub_yolo(COCO_NAMES)
    d = EdgeDetector().load()
    assert d.target_classes == [0, 1, 2, 3, 5, 7]
    assert d.class_names[5] == "bus"
    assert d.class_names[7] == "truck"


def test_a_model_without_names_keeps_the_defaults(stub_yolo):
    """A checkpoint with no taxonomy must not wipe the configured map."""
    stub_yolo({})
    d = EdgeDetector().load()
    assert d.class_names[0] == "person"
    assert d.target_classes == [0, 1, 2, 3, 5, 7]


def test_classes_ibvap_does_not_act_on_are_not_targeted(stub_yolo):
    """A custom model carrying extra classes must not have them targeted."""
    names = dict(IDD_NAMES)
    names[7] = "traffic light"
    names[8] = "animal"
    stub_yolo(names)
    d = EdgeDetector(model_path="custom.pt").load()

    assert 7 not in d.target_classes
    assert 8 not in d.target_classes
    assert 6 in d.target_classes


def test_detections_carry_the_adopted_class_name(stub_yolo):
    """End-to-end: a box at index 6 must surface as autorickshaw."""
    import numpy as np

    class Box:
        cls = [6]
        conf = [0.9]
        xyxy = [type("T", (), {"tolist": lambda self: [10.0, 20.0, 110.0, 140.0]})()]

    class Result:
        boxes = [Box()]

    stub_yolo(IDD_NAMES)
    d = EdgeDetector(model_path="models/weights/detection.pt").load()
    d.model = lambda *a, **k: [Result()]

    detections = d.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(detections) == 1
    assert detections[0].class_name == "autorickshaw"
    assert detections[0].class_id == 6
