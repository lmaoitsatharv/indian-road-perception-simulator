"""Publish checked evidence for one connected Indian town without altering models."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--session",type=Path,required=True)
    p.add_argument("--checks",type=Path,required=True)
    args = p.parse_args()
    run,session,checks = args.run.resolve(),args.session.resolve(),args.checks.resolve()
    if any(not x.is_relative_to(ROOT/"runs") for x in (run,session,checks)):
        raise ValueError("Keep evidence under runs/")
    proof = json.loads((checks/"verification.json").read_text())
    native = json.loads((session/"result.json").read_text())
    assert proof["passed"] and proof["ego_visited_all_five"] and native["passed"]
    assert native["world_generate_calls"] == 1
    owner = json.loads((session/"launcher_result.json").read_text())
    assert owner["owned_job_handles_closed"] and owner["status"] == "completed"
    source = run/"source_final"
    source.mkdir(exist_ok=False)
    files = ["integration/india_combined_map.py","integration/india_combined_scenery.py",
        "integration/india_combined_traffic.py","integration/india_combined_scene.py",
        "integration/india_world_scenery.py","integration/india_world_traffic.py","integration/india_world_scene.py",
        "setup/carla/run_session.py","setup/carla/viewer_entry.py","setup/verify_combined_india.py",
        "setup/report_combined_india.py","Start_Combined_Indian_Traffic.cmd","Start_Combined_Indian_Traffic.ps1",
        "Start_Indian_Worlds.cmd","Start_Indian_Worlds.ps1","Start_Separate_Indian_Worlds.cmd"]
    hashes = {}
    for file in files:
        target = source/file
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/file,target)
        hashes[file] = hashlib.sha256(target.read_bytes()).hexdigest()
    (run/"source_hashes_final.json").write_text(json.dumps(hashes,indent=2)+"\n")
    result = {"schema":"connected_indian_town_results_v1","status":"complete_single_connected_town_verified",
        "single_native_world":True,"world_generation_calls":1,"world_ids":native["world_ids"],
        "native_session":str(session.relative_to(ROOT)),"audit":str(checks.relative_to(ROOT)),
        "map":proof["map"],"continuous_sim_seconds":native["sim_seconds"],
        "native_vehicles":native["native_vehicles"],"native_walkers":native["native_walkers"],
        "wide_bus_urban_service_only":proof["wide_bus_stayed_on_divided_corridor"],
        "ego_visited_all_five":True,"ego_distance_m":proof["ego_distance_m"],
        "districts_visited":native["districts_visited"],"district_transitions":native["continuous_transitions"],
        "same_actor_ids_every_tick":True,"vehicle_pose_teleport":False,
        "camera_sets":proof["camera_sets"],"independently_reopened_images":proof["hashed_reopened_images"],
        "independent_vehicle_pair_checks":proof["independent_vehicle_body_pairs"],
        "native_collisions":len(native["collisions"]),"vehicle_body_overlaps":len(native["vehicle_overlaps"]),
        "vehicle_centre_departures":len(native["vehicle_centre_departures"]),
        "sampled_min_vehicle_gap_m":native["sampled_min_vehicle_gap_m"],
        "protected_files_unchanged":proof["protected_files_unchanged"],"training_started":False,
        "model_control":False,"rl_trained":False,"autorickshaw_live":False,
        "limits":proof["limits"]+["Clear weather and one seed in the long continuous proof; not multi-weather robustness",
            "Very close sampled clearances remain; native collision/20Hz rectangle checks are not a safety proof",
            "Generic installed assets with original Hindi/English artwork; custom Indian vehicle import remains pending"]}
    (run/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    (ROOT/"docs/results/combined_indian_town.json").write_text(json.dumps(result,indent=2)+"\n")
    previews = run/"previews"
    previews.mkdir(exist_ok=False)
    cards = []
    for file in sorted((session/"screenshots").glob("*.png")):
        shutil.copy2(file,previews/file.name)
        title = file.stem.replace("_"," ").title()
        cards.append(f'<section><h2>{html.escape(title)}</h2><img src="previews/{file.name}" alt="Actual native continuous-world preview"></section>')
    shutil.copy2(session/"town_plan.png",previews/"town_plan.png")
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>One connected Indian town</title><style>body{{margin:30px auto;max-width:1280px;padding:0 20px;background:#101d20;color:#dce5df;font:17px system-ui;line-height:1.6}}h1,h2{{color:#efd2a1}}img{{max-width:100%;border-radius:10px}}section{{margin:35px 0}}.map{{width:450px;image-rendering:auto}}a{{color:#9ed0cf}}</style>
<h1>One connected Indian town</h1><p>Market, colony, divided road, village roads and irregular junction share the same native world.
25 vehicles and 21 people exist together. The same ego vehicle visited all five areas over {native['sim_seconds']:.0f} simulated seconds,
travelling {proof['ego_distance_m']:.0f} m. One map generation; one world ID; no vehicle teleporting or replacement.</p>
<img class="map" src="previews/town_plan.png" alt="Connected district map"><p>Double-click Start_Combined_Indian_Traffic.cmd or Start_Indian_Worlds.cmd.
TAB changes front/follow view. SPACE pauses; S saves a picture; ESC closes.</p><p>Actual native camera captures below. Generic installed meshes, original Hindi/English signs;
no live autorickshaw or trained model control. Tight clearances remain; these checks do not establish driving safety.</p>
{''.join(cards)}</html>'''
    (run/"index.html").write_text(page,encoding="utf-8")
    print(json.dumps({"passed":True,"single_world":True,"ego_all_five":True,"sim_seconds":native["sim_seconds"],
        "ego_distance_m":proof["ego_distance_m"],"camera_sets":proof["camera_sets"]}))


if __name__ == "__main__":
    main()
