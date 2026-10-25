# One connected Indian town

Double-click **Start_Combined_Indian_Traffic.cmd** or **Start_Indian_Worlds.cmd**.
Both now open one native world containing the market, residential colony,
divided main road, village roads and irregular junction together. They are joined
by actual driving lanes. All traffic exists simultaneously; moving between areas
does not load a different map. The old separate-world selector is preserved as
`Start_Separate_Indian_Worlds.cmd`; individual `-Map` options still work.

## Connected layout and traffic

There are about 2.6 km of non-junction streets, seven junctions and 36 directed
lane connectors. A roughly 1.64 km main driving loop visits every district; a
residential shortcut loop and village/outskirts loop add alternative routes.
The main road has two lanes each way and a raised median, rural roads have
shoulders, and market/colony streets have footpaths and different widths. Rural,
market and colony streets include curved reference geometry.

Native LHT direction and 72 lane-centre joins passed offline checks; the largest
sampled join gap is about 2 cm. Both urban lanes connect to other roads. This is
an original illustrative layout, not a reconstruction of a real Indian town.

There are 25 native physics vehicles and 21 sidewalk pedestrians across the
world. Cars, scooters, motorcycles, cycles and delivery vans follow persistent
loops. The installed bus has a very wide native body, so it serves the divided
urban corridor and waits at a stop instead of entering the narrow village/market
roads. This is a route constraint on the real native asset, not a scaled-down
collision box. The same vehicle IDs cross district boundaries without live
teleports/replacement. Ground-truth body-gap rules and junction reservations
control traffic; no trained perception model supplies vehicle commands.

Hindi/English shops/signs, food carts, kiosks, bus stops, houses and palms are
distributed across the districts. Vehicle/building meshes are generic installed
assets. The autorickshaw is still source artwork only: no compatible Unreal
import/rig/cooking setup or registered native blueprint exists locally.

## Use

- TAB switches front/follow cameras.
- SPACE pauses/resumes; S saves a picture; ESC closes the owned simulation.
- The minimap shows all areas, traffic and ego position together.

Default duration is 900 simulation seconds, bounded by 1,800 wall seconds.
Building the whole town takes longer than a small scene. Low graphics is the
default. Preview is 5 Hz; synchronized evidence is saved at 1 Hz.

```powershell
.\Start_Combined_Indian_Traffic.ps1
.\Start_Combined_Indian_Traffic.ps1 -SimulationSeconds 300 -Weather overcast
```

Clear/overcast/wet/dusk are selectable; available options do not establish
weather robustness. Original small-neighbourhood and road-mask model launchers
remain separate.

## Evidence and limits

The completed continuous proof ran **650 simulation seconds / 13,000 physics
steps** in `runs/carla_indian_worlds_smoke_20261023T144946_280593Z_seed8101/`.
The same ego car travelled about **2.21 km**, visited every district and completed
the main loop without a new world load or vehicle replacement. One world ID and
all 25 vehicle identities persisted across the trace. Native collision events,
sampled vehicle overlaps, centre departures and route-direction errors were zero.
The bus stayed on the divided urban corridor.

Independent checks reopened/hashed **3,250 images in 650 synchronized sets**,
checked native road centres, recomputed vehicle rectangle overlaps and position
continuity at every physics step, and verified saved asset-label previews. All
11 protected source/model fingerprints were unchanged. This is one seed in clear
weather, not a general traffic/safety benchmark. The minimum sampled vehicle
gap was 0.273 m, so tight clearances remain despite zero recorded contact.

Review the native gallery: `runs/combined_india_20261023_01/index.html` (local artifact, supplied separately).,
[portable results](results/combined_indian_town.json), and the local independent
proof at `runs/combined_india_20261023_01/final_checks/verification.json`.
Source snapshots/fingerprints are in `runs/combined_india_20261023_01/source_final/`
and `source_hashes_final.json`. All owned native jobs and sensors closed cleanly.

The 75-second pilot completed with one native world generation, one unchanged
world ID, all 25 vehicle IDs retained, 21 walkers and no recorded vehicle
collisions/overlaps or vehicle-centre road departures. Independent checks
reopened/hashed 375 images in 75 synchronized camera sets and recomputed vehicle
rectangle overlaps/position continuity. This pilot covers early progress, not a
complete tour. The longer continuous drive is recorded in the completion report.

Retained development runs exposed a tangent-projection deadlock against
opposite-direction waiting traffic, followed by an actual narrow-road contact
when the oversized bus was allowed past that check. The following rule now
distinguishes opposing/adjacent lanes and the bus uses the wide corridor. Those
failed runs are preserved separately and are not the completed continuity proof.

RGB, raw semantics, raw instances, encoded depth and follow RGB share native
frame/time. Raw labels are preserved. Exact owned instance IDs produce separate
corrected asset-mask previews; unknown pixels and sensor-edge disagreements are
ignored. These are inspected development captures, not a newly split dataset or
the complete depth/map road-edge reference audit.

This map/capture evaluation performed no model training, checkpoint changes, real deployment or RL. Tight
sampled vehicle gaps can still occur; bounded checks do not establish driving
safety, calibrated human behaviour or real-world performance.
