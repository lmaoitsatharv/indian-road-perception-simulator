# Five Indian road worlds — 23 October 2026

Five separate native CARLA/OpenDRIVE worlds are built and checked. This completes
the map upgrade and a first scenery/vehicle pass. **Custom Indian vehicle import
is incomplete:** the autorickshaw is source artwork only, not live traffic.

## Launch

The later [combined-town upgrade](COMBINED_INDIAN_TOWN.md) makes
**Start_Indian_Worlds.cmd** open all areas together in one connected map.
The separate-world selector described here is preserved as
**Start_Separate_Indian_Worlds.cmd**.

Double-click **Start_Separate_Indian_Worlds.cmd**. Choose market, residential, divided main
road, rural, irregular junction or a tour of all five. Default duration is 60
simulation seconds per world. SPACE pauses/resumes, S saves a picture, ESC closes
the owned simulator. Allow startup to finish. Low graphics is the default.

Optional PowerShell:

```powershell
.\Start_Indian_Worlds.ps1 -Map market_street
.\Start_Indian_Worlds.ps1 -Map residential_colony
.\Start_Indian_Worlds.ps1 -Map divided_main_road
.\Start_Indian_Worlds.ps1 -Map rural_road
.\Start_Indian_Worlds.ps1 -Map irregular_junction
.\Start_Indian_Worlds.ps1 -Map all -Check
```

Clear/overcast/wet/dusk and Low/Epic are selectable. The final checks used clear
weather only. The old neighbourhood and live road-mask preview remain separate.
New worlds have no perception-model overlay or model-controlled vehicles.

## Roads

| World | Actual native geometry | Driving-lane width |
|---|---|---|
| Market | 260 m; two bends; gradual widening; 1.8 m footpaths; closer shops | 2.85–3.10 m |
| Residential | Three connected approaches; skewed, narrower side lane; 2.0 m footpaths | 2.65–3.10 m |
| Divided main road | 360 m; gentle curve; two lanes each way; 2.2 m raised median; 2.2 m footpaths | 3.25 m |
| Rural | 390 m; three curved sections; changing width; 1.35 m shoulders; no urban footpaths | 2.60–2.75 m |
| Irregular junction | Four skewed approaches; twelve turn connectors; varied lengths/widths; 1.5 m footpaths | 2.70–3.65 m |

All five map hashes differ. Native checks verify explicit left-hand lane rules,
lane travel direction, connected route positions/headings, junction branches and
median/shoulder definitions. No stock right-hand town is relabelled Indian.

## Assets and remaining import work

Original Hindi/English artwork covers kirana, chai/snacks, vegetables, pharmacy,
cycle repair, dairy, bus stops, keep-left, village/school and speed signs.
Hindi uses native Windows shaping. A UV-grid probe measured the installed banner
mesh's visible texture area; the failed tiled-floor board attempt is retained.
Kiosks, food carts, bus shelters, houses/apartments and palms use installed meshes.
Facades use actual native bounds/setbacks. Placement checks sample all three
bounds axes, including rotated boards. Asphalt is sampled inside native driving
geometry; its thin appearance collision is disabled while road collision remains.

Each world has seven native physics vehicles: hatchbacks, Vespa scooter, Yamaha
motorcycle, bicycle, Mitsubishi minibus and Mercedes delivery van. Urban worlds
also have four sidewalk walkers. Vehicles/buildings are **generic installed
meshes**, not newly imported Indian makes or calibrated local fleets. The delivery
livery is source artwork only and is not applied to the van body.

An original unbranded autorickshaw OBJ/MTL and three-wheel/collision manifest is
staged in `integration/indian_world_assets/autorickshaw_source/`. Its manifest
blocks runtime and training use. Native inventory has no autorickshaw blueprint;
a CARLA-compatible Unreal source/editor/cooking setup is not installed. Rigging,
cooking, registration, native wheel/collision and semantic tests remain required.
A van/car has not been relabelled autorickshaw. See CARLA's
[vehicle authoring workflow](https://carla.readthedocs.io/en/0.9.14/tuto_A_add_vehicle/).

## Evidence and limits

Final native session:
`runs/carla_indian_worlds_smoke_20261023T134059_973058Z_seed7101/`.
Each world ran 60 seconds / 1,200 physics steps, clear weather, seed 7101.

| World | Native collisions | Sampled vehicle overlaps | Vehicle-centre road departures | Route-direction errors | Smallest sampled vehicle gap |
|---|---:|---:|---:|---:|---:|
| Market | 0 | 0 | 0 | 0 | 1.106 m |
| Residential | 0 | 0 | 0 | 0 | 0.138 m |
| Divided | 0 | 0 | 0 | 0 | 0.843 m |
| Rural | 0 | 0 | 0 | 0 | 1.271 m |
| Irregular junction | 0 | 0 | 0 | 0 | 0.195 m |

Tight junction clearances remain. Zero recorded contact is not comfortable
spacing or a safety proof. Vehicle-body checks sample at 20 Hz; a centre check is
not a whole-body road-boundary guarantee. New pedestrian interactions were not
validated. Some queues stop before route completion. Driving uses ground truth
and explicit rules, not camera predictions, calibrated human behaviour or RL.

The first residential attempt retained two van/kerb contacts. New-world-only
shorter steering lookahead and slower turns removed recorded contact in the
final run. Route-direction checking uses the connected native route because
overlapping junction/approach surfaces can select an opposite nearest lane.
Failures and that original ambiguity are retained; old `india_behavior.py` is
unchanged. One seed per final world does not establish general robustness.

Front RGB, semantic, instance and encoded-depth sensors are frame/time aligned;
overview RGB shares that frame. Preview processing is 5 Hz; saved capture is 1 Hz.
There are **300 capture sets / 1,500 images**. Independent checks reopened/hashed
every image, checked timestamps, decoded depth, compared semantic/instance tags,
checked actor displacement and verified all five maps/routes.

A later visual inspection caught upside-down banner artwork. Only its atlas
rotation changed; geometry, physics and control code stayed unchanged. All five
worlds then passed an additional 14-second native material/camera check in
`runs/carla_indian_worlds_smoke_20261023T140834_577890Z_seed7101/`. The gallery
uses those upright previews. Its 70 additional capture sets / 350 images and
asset masks also passed independent checks. Both evidence sets are preserved.

Raw native tags remain unchanged. Banner assets render as Other; concrete
appearance renders as Building. Separate asset-corrected semantic previews use
exact owned instance IDs: signs → traffic sign, shop boards → building, asphalt
→ road, ground → terrain, median → non-road sidewalk/median. Unrelated known
classes remain unchanged; unresolved procedural pixels and sensor disagreement
are ignored. Every mask was reopened and visible owned-pixel mappings checked.

These are inspected development previews, **not** the full depth/map road-edge
audit, detector-box export, new 5,000-image dataset or training approval. The old
5,000-pair dataset and trained models are unchanged. Old high simulation IoU does
not automatically apply to these new worlds.

## Review files

- Native camera previews beside lane plans: `runs/indian_worlds_20261023_01/index.html` (local artifact, supplied separately)..
- [Portable results and completion flags](results/indian_worlds.json).
- Independent checks: `runs/indian_worlds_20261023_01/final_checks/verification.json`.
- Label checks/masks: `runs/indian_worlds_20261023_01/asset_label_previews/`.
- Latest visual/mask checks: `runs/indian_worlds_20261023_01/visual_checks/`, `asset_label_previews_visual/`.
- Final source snapshots/hashes: `runs/indian_worlds_20261023_01/source_final/`, `source_hashes_final.json`.
- The earlier source snapshot and failed visual attempts remain preserved.

Owned processes/sensors closed cleanly. Original neighbourhood geometry/driver,
both road checkpoints and real inference have unchanged before/after hashes.
No training, real-model replacement or RL occurred. Following work is authentic
vehicle import, better junction clearance and a full label audit/map-family split
before new dataset generation or retraining.
