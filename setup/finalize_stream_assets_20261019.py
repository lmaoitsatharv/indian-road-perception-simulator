"""Final local-file checks and an explicitly unreviewed annotation import."""
import json
from html.parser import HTMLParser
from pathlib import Path
import shutil
import sys
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from idd_inference import ROOT, file_sha256, write_json_atomic
from road_video_labels import score_pack


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ("href", "src") and value)


def main():
    run = ROOT / "runs/real_stream_20261019_01"
    pack = ROOT / "runs/real_video_labels_20261019_01"
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    check = ROOT / "runs/real_label_import_draftcheck_20261019_01"
    check.mkdir(exist_ok=False)
    draft = {"schema": "road_annotations_v1", "pack_id": manifest["pack_id"],
             "pack_manifest_sha256": file_sha256(pack / "manifest.json"),
             "rubric_id": manifest["rubric_id"], "reviewer": "UNREVIEWED EMPTY FORMAT CHECK",
             "frames": [], "note": "No human labels. This checks that empty data cannot publish accuracy."}
    write_json_atomic(check / "empty_draft.json", draft)
    result = score_pack(pack, check / "empty_draft.json", check / "no_labels_result")
    assert result["primary_score"] is None and result["reviewed_scoring_frames"] == 0
    assert result["state"] == "awaiting_all_scoring_labels"
    (check / "README.md").write_text("# Empty-format check only\n\nNo human annotations were supplied or invented. The actual fixed pack was validated with an empty draft, and its primary score correctly stayed null. This folder is verification evidence, not a video accuracy result.\n", encoding="utf-8")
    files = [run / "index.html", pack / "index.html", ROOT / "media/real_driving_20261017/index.html"]
    checked = 0
    for path in files:
        parser = Links()
        parser.feed(path.read_text(encoding="utf-8"))
        for link in parser.links:
            if urlsplit(link).scheme or link.startswith("#"):
                continue
            assert (path.parent / unquote(link.split("#")[0])).is_file(), (str(path), link)
            checked += 1
    for name in ["road_video_labels.py", "setup/road_label_editor_template.html", "setup/report_road_stream_20261019.py"]:
        shutil.copy2(ROOT / name, run / "source_snapshot" / Path(name).name)
    for name in ["road_video_labels.py", "setup/road_label_editor_template.html"]:
        shutil.copy2(ROOT / name, pack / "source_snapshot" / Path(name).name)
    status = json.loads((run / "run_status.json").read_text(encoding="utf-8"))
    audit = json.loads((run / "independent_verification.json").read_text(encoding="utf-8"))
    labels = json.loads((pack / "label_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed" and audit["state"] == "passed"
    assert labels["state"] == "awaiting_human_labels" and labels["accuracy"] is None
    assert all(r["decoder_thread_stopped"] for r in status["results"])
    for relative, sha in status["preservation"]["protected_files_after"].items():
        assert file_sha256(ROOT / relative) == sha
    evidence = {"state": "passed", "local_html_links_checked": checked,
                "empty_real_pack_import_publishes_no_accuracy": True,
                "real_label_status": labels["state"], "real_accuracy": None,
                "stream_results": audit["modes"], "all_owned_decoder_threads_stopped": True,
                "protected_files_still_unchanged": True,
                "actual_editor_browser_interactions_verified": False}
    write_json_atomic(run / "completion_checks.json", evidence)
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
