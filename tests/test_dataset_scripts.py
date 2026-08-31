"""Dataset preparation — annotation parsing, box math, and deduplication.

These scripts run once per dataset and then their output trains a model for
hours, so a silent conversion bug is expensive: it surfaces as bad mAP,
which looks like a model problem. The box-normalisation tests pin exact
values for that reason.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import exdark_to_yolo
import idd_to_yolo
import plates_to_yolo


def write_image(path: Path, height=300, width=400, value=128):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((height, width, 3), value, dtype=np.uint8))
    return path


# --- shared box math ----------------------------------------------------

def test_idd_box_normalisation_is_exact():
    # Box (100,150)-(200,190) in a 400x300 image.
    line = idd_to_yolo.to_yolo_line("car", 100, 150, 200, 190, 400, 300)
    cls, cx, cy, w, h = line.split()
    assert cls == str(idd_to_yolo.CLASS_TO_IDX["car"])
    assert float(cx) == pytest.approx(0.375)      # 150 / 400
    assert float(cy) == pytest.approx(0.566667, abs=1e-5)   # 170 / 300
    assert float(w) == pytest.approx(0.25)        # 100 / 400
    assert float(h) == pytest.approx(0.133333, abs=1e-5)    # 40 / 300


def test_plate_boxes_are_always_class_zero():
    """The localizer is single-class by design; class index must never drift."""
    line = plates_to_yolo.to_yolo_line(100, 150, 200, 190, 400, 300)
    assert line.split()[0] == "0"


def test_boxes_overshooting_the_frame_are_clamped():
    line = idd_to_yolo.to_yolo_line("person", -50, -50, 500, 400, 400, 300)
    values = [float(v) for v in line.split()[1:]]
    assert all(0.0 <= v <= 1.0 for v in values)


# --- IDD deduplication --------------------------------------------------

def test_identical_frames_hash_the_same(tmp_path):
    a = write_image(tmp_path / "a.jpg", value=100)
    b = write_image(tmp_path / "b.jpg", value=100)
    assert idd_to_yolo._hamming(idd_to_yolo._dhash(a), idd_to_yolo._dhash(b)) == 0


def test_deduplication_drops_near_identical_dashcam_frames(tmp_path):
    """The reason this exists: IDD at 30fps is mostly redundant frames."""
    frames = []
    base = np.full((120, 160, 3), 60, dtype=np.uint8)
    base[20:60, 30:90] = 200                      # one distinct shape
    for i in range(5):
        path = tmp_path / f"seq_{i:03d}.jpg"
        cv2.imwrite(str(path), base)              # five identical frames
        frames.append(path)

    kept = idd_to_yolo.deduplicate(frames, distance=5)
    assert len(kept) == 1
    assert kept[0] == frames[0]                   # the first is the one kept


def test_deduplication_keeps_genuinely_different_scenes(tmp_path):
    frames = []
    for i in range(4):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        # Move a large block far enough that the hash genuinely differs.
        image[:, i * 40:(i + 1) * 40] = 255
        path = tmp_path / f"scene_{i}.jpg"
        cv2.imwrite(str(path), image)
        frames.append(path)

    assert len(idd_to_yolo.deduplicate(frames, distance=5)) == len(frames)


def test_deduplication_compares_against_the_last_kept_frame(tmp_path):
    """A slow pan must not ratchet through in sub-threshold steps."""
    frames = []
    base = np.full((120, 160, 3), 60, dtype=np.uint8)
    base[20:60, 30:90] = 200
    for i in range(6):
        cv2.imwrite(str(tmp_path / f"pan_{i}.jpg"), base)
        frames.append(tmp_path / f"pan_{i}.jpg")

    assert len(idd_to_yolo.deduplicate(frames, distance=5)) == 1


def test_unreadable_images_are_skipped_not_fatal(tmp_path):
    good = write_image(tmp_path / "good.jpg")
    broken = tmp_path / "broken.jpg"
    broken.write_text("not an image")

    assert idd_to_yolo._dhash(broken) is None
    assert idd_to_yolo.deduplicate([broken, good], distance=5) == [good]


# --- VOC parsing --------------------------------------------------------

VOC_XML = """<annotation>
  <size><width>400</width><height>300</height></size>
  <object><name>licence</name>
    <bndbox><xmin>100</xmin><ymin>150</ymin><xmax>200</xmax><ymax>190</ymax></bndbox>
  </object>
</annotation>"""


def test_voc_parsing_returns_size_and_boxes(tmp_path):
    xml_path = tmp_path / "car.xml"
    xml_path.write_text(VOC_XML)
    width, height, boxes = plates_to_yolo.parse_voc(xml_path)
    assert (width, height) == (400, 300)
    assert boxes == [(100.0, 150.0, 200.0, 190.0)]


@pytest.mark.parametrize("label", ["licence", "license", "license_plate", "plate"])
def test_voc_accepts_the_label_spellings_mirrors_use(tmp_path, label):
    """Kaggle mirrors disagree on the label name; all must convert."""
    xml_path = tmp_path / f"{label}.xml"
    xml_path.write_text(VOC_XML.replace("licence", label))
    assert plates_to_yolo.parse_voc(xml_path)[2]


def test_voc_drops_non_plate_objects(tmp_path):
    xml_path = tmp_path / "mixed.xml"
    xml_path.write_text(VOC_XML.replace("licence", "traffic_light"))
    assert plates_to_yolo.parse_voc(xml_path)[2] == []


def test_voc_with_a_degenerate_box_is_dropped(tmp_path):
    xml_path = tmp_path / "bad.xml"
    xml_path.write_text(VOC_XML.replace("<xmax>200</xmax>", "<xmax>100</xmax>"))
    assert plates_to_yolo.parse_voc(xml_path)[2] == []


# --- UFPR parsing -------------------------------------------------------

UFPR_TXT = """camera: 1
position_vehicle: 10 10 100 100
type: car
plate: ABC1D23
position_plate: 100 150 100 40
char 1: 1 1 5 5
"""


def test_ufpr_converts_corner_plus_size_to_corner_plus_corner(tmp_path):
    """UFPR gives x y w h, not xmin ymin xmax ymax — the easy bug to write."""
    txt_path = tmp_path / "f0.txt"
    txt_path.write_text(UFPR_TXT)
    boxes, plate = plates_to_yolo.parse_ufpr(txt_path)
    assert boxes == [(100, 150, 200, 190)]
    assert plate == "ABC1D23"


def test_ufpr_without_a_plate_string_still_yields_a_box(tmp_path):
    txt_path = tmp_path / "f1.txt"
    txt_path.write_text(UFPR_TXT.replace("plate: ABC1D23\n", ""))
    boxes, plate = plates_to_yolo.parse_ufpr(txt_path)
    assert boxes and plate is None


# --- ExDark parsing -----------------------------------------------------

def test_exdark_annotation_maps_and_normalises(tmp_path):
    ann = tmp_path / "car_0.jpg.txt"
    ann.write_text("% comment\nCar 20 30 60 40 0 0 0 0\nPeople 5 5 20 50 0 0 0 0\n")

    lines = exdark_to_yolo.parse_exdark_annotation(ann, img_w=200, img_h=120)
    assert len(lines) == 2

    car = lines[0].split()
    assert car[0] == str(exdark_to_yolo.CLASS_TO_IDX["car"])
    assert float(car[1]) == pytest.approx(0.25)        # (20 + 30) / 200
    assert float(car[2]) == pytest.approx(0.416667, abs=1e-5)   # (30 + 20) / 120


def test_exdark_drops_classes_ibvap_does_not_act_on(tmp_path):
    ann = tmp_path / "bottle.jpg.txt"
    ann.write_text("% comment\nBottle 10 10 20 20 0 0 0 0\n")
    assert exdark_to_yolo.parse_exdark_annotation(ann, 200, 120) == []


def test_exdark_clamps_boxes_running_past_the_edge(tmp_path):
    ann = tmp_path / "edge.jpg.txt"
    ann.write_text("% comment\nCar 180 100 200 200 0 0 0 0\n")
    lines = exdark_to_yolo.parse_exdark_annotation(ann, 200, 120)
    assert lines
    values = [float(v) for v in lines[0].split()[1:]]
    assert all(0.0 <= v <= 1.0 for v in values)


# --- synthetic low-light ------------------------------------------------

def test_darkening_reduces_brightness_without_changing_shape():
    import random
    bright = np.full((60, 80, 3), 200, dtype=np.uint8)
    dark = exdark_to_yolo.darken_image(bright, random.Random(0))

    assert dark.shape == bright.shape
    assert dark.dtype == np.uint8
    assert dark.mean() < bright.mean() * 0.8


def test_darkening_only_touches_the_train_split(tmp_path):
    """Val must stay genuinely dark, never synthetically darkened."""
    for split in ("train", "val"):
        write_image(tmp_path / "images" / split / f"idd_{split}.jpg")
        (tmp_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        (tmp_path / "labels" / split / f"idd_{split}.txt").write_text("2 0.5 0.5 0.2 0.2\n")

    exdark_to_yolo.darken_split(tmp_path, fraction=1.0)

    train_files = {p.name for p in (tmp_path / "images" / "train").iterdir()}
    val_files = {p.name for p in (tmp_path / "images" / "val").iterdir()}
    assert any(name.startswith("dark_") for name in train_files)
    assert not any(name.startswith("dark_") for name in val_files)


def test_darkening_is_not_applied_to_already_dark_images(tmp_path):
    """ExDark photographs are real low-light; darkening them again is wrong."""
    write_image(tmp_path / "images" / "train" / "exdark_a.jpg")
    (tmp_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels" / "train" / "exdark_a.txt").write_text("2 0.5 0.5 0.2 0.2\n")

    with pytest.raises(SystemExit):
        exdark_to_yolo.darken_split(tmp_path, fraction=1.0)


# --- OCR ground-truth template ------------------------------------------
#
# The end-to-end exact-match number is the one thing the localizer A/B
# (F1 0.194 -> 0.915) does not answer: it scores whether the box was found,
# not whether the text was read. Measuring that needs 91 hand-transcribed
# plates, so the template these tests guard is the artifact that manual work
# lands in — losing it silently would mean transcribing them twice.

def _seed_images(images_dir: Path, names):
    images_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        cv2.imwrite(str(images_dir / name), np.zeros((40, 80, 3), dtype=np.uint8))


def test_template_has_one_blank_entry_per_image(tmp_path):
    import evaluate

    images = tmp_path / "images"
    _seed_images(images, ["a.jpg", "b.png"])
    out = tmp_path / "plates_val.json"

    blanks = evaluate._write_gt_template(images, out)

    assert blanks == 2
    assert json.loads(out.read_text()) == {"a.jpg": "", "b.png": ""}


def test_regenerating_the_template_preserves_transcribed_plates(tmp_path):
    """The whole point: a re-run must not destroy 45 minutes of typing."""
    import evaluate

    images = tmp_path / "images"
    _seed_images(images, ["a.jpg"])
    out = tmp_path / "plates_val.json"
    out.write_text(json.dumps({"a.jpg": "MH12AB1234"}))

    _seed_images(images, ["b.jpg"])
    blanks = evaluate._write_gt_template(images, out)

    written = json.loads(out.read_text())
    assert written["a.jpg"] == "MH12AB1234", "existing label was overwritten"
    assert written["b.jpg"] == ""
    assert blanks == 1


def test_template_ignores_non_image_files(tmp_path):
    import evaluate

    images = tmp_path / "images"
    _seed_images(images, ["a.jpg"])
    (images / "notes.txt").write_text("scratch")
    out = tmp_path / "plates_val.json"

    evaluate._write_gt_template(images, out)

    assert "notes.txt" not in json.loads(out.read_text())
