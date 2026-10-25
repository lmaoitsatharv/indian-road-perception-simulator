"""Check sampling probabilities and reject mismatched refinement provenance."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detection_refinement import sampling_weights
from idd_detection_data import save_json
from train_detection import readiness


def main(output, recipe_path):
    import numpy as np
    output.mkdir(exist_ok=False)
    names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"]
    groups = ["same", "same", "animal", "bike", "negative"]
    labels = [{"cls": np.asarray(values).reshape(-1, 1)} for values in ([0], [0], [8], [4], [])]
    rows = [{"split": "train", "prepared_image": "images/train/" + name, "pixels_sha256": group}
            for name, group in zip(names, groups)]
    rows.append({"split": "val", "prepared_image": "images/val/never_train.jpg", "pixels_sha256": "test"})
    weights = sampling_weights(labels, names, rows, {"4": 2, "8": 4, "9": 2})
    np.testing.assert_array_equal(weights, [0.5, 0.5, 4, 2, 1])
    assert weights[:2].sum() == 1 and len(weights) == len(names) and (weights > 0).all()
    for boost in (0, 5, float("nan")):
        try:
            sampling_weights(labels, names, rows, {"8": boost})
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid weighting was accepted")
    recipe = json.loads(recipe_path.read_text())
    ready = readiness(recipe_path=recipe_path, inspect_files=False)
    assert ready["ready_for_explicit_training_start"], ready["waiting_for"]
    provenance = json.loads((ROOT / recipe["initial_weights_metadata"]).read_text())
    bad_provenance = {**provenance, "verification_sha256": "0" * 64}
    save_json(output / "bad_provenance.json", bad_provenance)
    bad_recipe = {**recipe, "initial_weights_metadata": (output / "bad_provenance.json").relative_to(ROOT).as_posix()}
    save_json(output / "bad_recipe.json", bad_recipe)
    assert not readiness(recipe_path=output / "bad_recipe.json", inspect_files=False)["ready_for_explicit_training_start"]
    if recipe.get("training_view_verification"):
        changed_view = {**recipe,"training_view_verification_sha256":"0"*64}
        save_json(output/"bad_view_recipe.json",changed_view)
        assert not readiness(recipe_path=output/"bad_view_recipe.json",inspect_files=False)["ready_for_explicit_training_start"]
    assert readiness(inspect_files=False)["ready_for_explicit_training_start"], "The original recipe must still work"
    save_json(output / "verification.json", {"passed": True, "training_started": False,
        "checks": ["rare_image_weights_are_bounded_and_expected", "duplicate_pixel_group_base_weights_sum_to_one",
            "all_original_training_identities_keep_positive_probability", "validation_rows_never_enter_weighted_training",
            "zero_excessive_and_nan_boosts_rejected", "actual_refinement_metadata_ready",
            "changed_checkpoint_verification_hash_rejected", "original_coco_recipe_still_ready"],
        "clean_training_view_guard_checked":bool(recipe.get("training_view_verification")),
        "full_file_audit": "trainer repeats before creating a run", "model_control": False})
    print((output / "verification.json").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe",type=Path,default=ROOT/"detection_refinement_recipe.json")
    args=parser.parse_args()
    main(args.output.resolve(),args.recipe.resolve())
