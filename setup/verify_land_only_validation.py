"""Independent checks for <v4>/val_land against the unchanged <v4>/labels/val (read-only)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "dataset/idd117k_detection_v4_landanimals"
LAND = V4 / "val_land"


def lines(p):
    return [l.strip() for l in open(p) if l.strip()]


ok, notes = True, {}
imgs = {p.stem for p in (LAND / "images").glob("*.jpg")}
labs = {p.stem for p in (LAND / "labels").glob("*.txt")}
orig = {p.stem for p in (V4 / "labels/val").glob("*.txt")}
excluded = set(json.load(open(V4 / "provenance/val_land_excluded.json")))
checks = {
    "images and labels pair exactly": imgs == labs,
    "val_land + excluded = original validation": labs | excluded == orig and not (labs & excluded),
    "original validation untouched (17,675 labels)": len(orig) == 17675,
}
non_animal_changed, animal_added, animal = 0, 0, 0
for s in labs:
    a, b = lines(LAND / "labels" / (s + ".txt")), lines(V4 / "labels/val" / (s + ".txt"))
    if [l for l in a if not l.startswith("8 ")] != [l for l in b if not l.startswith("8 ")]:
        non_animal_changed += 1
    if not set(l for l in a if l.startswith("8 ")) <= set(l for l in b if l.startswith("8 ")):
        animal_added += 1
    animal += sum(1 for l in a if l.startswith("8 "))
checks["non-animal labels identical to original"] = non_animal_changed == 0
checks["animal boxes only removed, never added or moved"] = animal_added == 0
summary = json.load(open(V4 / "provenance/val_land_summary.json"))
checks["animal box count matches build summary"] = animal == summary["animal_boxes_in_val_land"]
for name, passed in checks.items():
    print(("PASS " if passed else "FAIL ") + name)
    ok &= passed
print("val_land images", len(labs), "| excluded", len(excluded), "| animal boxes", animal)
(V4 / "provenance/val_land_verification.json").write_text(json.dumps({"checks": checks, "all_passed": ok, "val_land_images": len(labs),
                                                                      "animal_boxes": animal}, indent=1))
print("ALL PASSED" if ok else "SOME CHECKS FAILED")
