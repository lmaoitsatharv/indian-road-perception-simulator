"""Package checked native Indian-world previews, map plans and limitations."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import sys

import carla
import numpy as np
import pygame

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"integration"))
from india_world_maps import PROFILES,build_world


def plan(key,target):
    native = carla.Map("preview_"+key,build_world(key))
    waypoints = native.generate_waypoints(.8)
    coords = np.array([[w.transform.location.x,w.transform.location.y] for w in waypoints])
    low,high = coords.min(0)-12,coords.max(0)+12
    scale = min(590/(high[0]-low[0]),290/(high[1]-low[1]))
    centre = (high+low)/2
    surface = pygame.Surface((620,340))
    surface.fill((21,35,37))
    def xy(x,y):
        return (int(310+(x-centre[0])*scale),int(170+(y-centre[1])*scale))
    for wp in waypoints:
        t = wp.transform
        forward = t.get_forward_vector()
        right = t.get_right_vector()
        points = [xy(t.location.x+u*forward.x+v*right.x,t.location.y+u*forward.y+v*right.y)
            for u,v in ((-.60,-wp.lane_width/2),(.60,-wp.lane_width/2),(.60,wp.lane_width/2),(-.60,wp.lane_width/2))]
        pygame.draw.polygon(surface,(132,149,146) if not wp.is_junction else (203,160,88),points)
    for wp in native.generate_waypoints(24.):
        if wp.is_junction:
            continue
        t = wp.transform
        yaw = math.radians(t.rotation.yaw)
        points = [xy(t.location.x+a*math.cos(yaw)-b*math.sin(yaw),t.location.y+a*math.sin(yaw)+b*math.cos(yaw))
            for a,b in ((2,0),(-1.2,1.1),(-1.2,-1.1))]
        pygame.draw.polygon(surface,(250,224,167),points)
    pygame.image.save(surface,str(target))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--session",type=Path,required=True)
    p.add_argument("--visual-session",type=Path,help="Later material-only native check; preserve original driving evidence")
    args = p.parse_args()
    run,session = args.run.resolve(),args.session.resolve()
    if not run.is_relative_to(ROOT/"runs") or not session.is_relative_to(ROOT/"runs"):
        raise ValueError("Keep evidence inside runs/")
    proof = json.loads((run/"final_checks/verification.json").read_text())
    labels = json.loads((run/"asset_label_previews/verification.json").read_text())
    native = json.loads((session/"result.json").read_text())
    assert proof["passed"] and labels["passed"] and native["passed"]
    visual = args.visual_session.resolve() if args.visual_session else session
    visual_native = json.loads((visual/"result.json").read_text())
    if not visual.is_relative_to(ROOT/"runs") or not visual_native["passed"] or len(visual_native["worlds"]) != 5:
        raise ValueError("All five visual checks must pass")
    visual_checks = json.loads((run/"visual_checks/verification.json").read_text()) if args.visual_session else None
    visual_labels = json.loads((run/"asset_label_previews_visual/verification.json").read_text()) if args.visual_session else None
    if args.visual_session:
        assert visual_checks["passed"] and visual_labels["passed"]
    pygame.init()
    (run/"previews").mkdir(exist_ok=True)
    for key in PROFILES:
        plan(key,run/"previews"/(key+"_map.png"))
        shutil.copy2(visual/key/"preview_0240.png",run/"previews"/(key+"_native.png"))
    pygame.quit()
    source = run/("source_final" if args.visual_session else "source")
    source.mkdir(exist_ok=False)
    files = ["integration/india_world_maps.py","integration/india_world_scenery.py","integration/india_world_scene.py",
        "integration/india_world_traffic.py","integration/india_world_asset_probe.py","setup/build_indian_world_assets.py",
        "setup/render_indian_sign_text.ps1","setup/verify_indian_worlds.py","setup/prepare_indian_world_labels.py",
        "setup/report_indian_worlds.py","setup/carla/run_session.py","setup/carla/viewer_entry.py",
        "Start_Indian_Worlds.cmd","Start_Indian_Worlds.ps1"]
    hashes = {}
    for name in files:
        target = source/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    for path in (ROOT/"integration/indian_world_assets").rglob("*"):
        if path.is_file():
            name = str(path.relative_to(ROOT))
            target = source/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
            hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    (run/("source_hashes_final.json" if args.visual_session else "source_hashes.json")).write_text(json.dumps(hashes,indent=2)+"\n")
    results = {"schema":"indian_worlds_results_v1","status":"five_native_worlds_verified_custom_vehicle_import_pending",
        "five_native_worlds_passed":True,"full_requested_asset_scope_complete":False,
        "worlds":native["worlds"],"native_checks":proof["native"],"map_checks":proof["maps"],
        "label_checks":labels["worlds"],"protected_files_unchanged":proof["protected_files_unchanged"],
        "camera_preview_hz":5,"saved_synchronized_capture_hz":1,"development_capture_frames":300,
        "native_session":str(session.relative_to(ROOT)),"run":str(run.relative_to(ROOT)),
        "upright_material_check_session":str(visual.relative_to(ROOT)),
        "upright_material_check_passed":visual_native["passed"],
        "upright_material_check_capture_sets":sum(r["paired_frames"] for r in visual_native["worlds"]),
        "upright_material_independent_checks":visual_checks,
        "upright_material_label_checks":visual_labels,
        "not_trained_or_replaced":True,"models_control_vehicles":False,
        "autorickshaw":{"status":"original_OBJ_MTL_and_three_wheel_manifest_prepared; not imported",
            "runtime_allowed":False,"training_allowed":False,
            "blocker":"Packaged CARLA has no autorickshaw blueprint; CARLA-compatible Unreal source/editor and cooking toolchain not installed"},
        "limits":["Illustrative Indian-style roads and original Hindi/English textures; generic installed architecture/vehicle meshes",
            "One seed and clear weather for each final 60s native check; not a calibrated Indian traffic benchmark",
            "Very close sampled vehicle clearances remain: minimum 0.138m residential and 0.195m irregular junction",
            "Native collision plus 20Hz vehicle-body checks are not continuous safety proof; no new pedestrian interaction validation",
            "Saved asset-corrected masks are development previews, not the full geometry/depth road-label audit or a new dataset split",
            "Rigid custom sign/banner meshes retain native raw class Other; exact instance-only correction is saved separately",
            "Detailed custom Indian bus/scooter/van meshes are not newly authored or imported"],
        "retained_attempts":["131823 controller tuple-unpacking failure",
            "131926 conservative tree-bounds failure after market run",
            "132240 material candidate probe", "132718 horizontal banner/spawn failure",
            "132931 first all-map run, residential kerb touches and nearest-lane junction direction ambiguity",
            "133725 banner UV-grid calibration", "134059 final all-map 60s verification",
            "Later all-map upright-banner visual capture; geometry/physics/controller unchanged"]}
    (run/"results.json").write_text(json.dumps(results,indent=2)+"\n")
    (ROOT/"docs/results/indian_worlds.json").write_text(json.dumps(results,indent=2)+"\n")
    rows = []
    for key in PROFILES:
        rows.append(f'<section><h2>{PROFILES[key].title}</h2><div class="pair"><img src="previews/{key}_native.png" alt="Actual native camera preview"><img src="previews/{key}_map.png" alt="Native lane geometry, arrows show LHT"></div></section>')
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Five Indian worlds — native evidence</title><style>body{font:16px system-ui;background:#111c20;color:#e3eae6;margin:30px auto;max-width:1500px;padding:0 20px}h1,h2{color:#edd2a0}section{margin:35px 0}.pair{display:grid;grid-template-columns:1.6fr 1fr;gap:16px}.pair img{width:100%;align-self:center;border-radius:8px}p{max-width:1000px;line-height:1.6}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style><h1>Five Indian road worlds</h1><p>Actual native CARLA previews beside independently parsed native lane plans. Driving checks: five 60-second scenes, 300 synchronized saved capture sets. Latest previews are from the subsequent 14-second checks of corrected upright artwork (70 additional sets). No model training or vehicle control by a model.</p><p>Generic installed vehicles and architecture with original Hindi/English artwork. The autorickshaw is source artwork only and is not in live traffic. Tight sampled clearances remain; these checks do not establish driving safety.</p>'+''.join(rows)+'</html>'
    (run/"index.html").write_text(html,encoding="utf-8")
    print(json.dumps({"native_worlds":5,"capture_sets":300,"source_files":len(hashes),"autorickshaw_live":False}))


if __name__ == "__main__":
    main()
