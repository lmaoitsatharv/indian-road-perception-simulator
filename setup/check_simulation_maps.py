"""Query native CARLA maps for LHT topology and all planned footpath samples."""
import json
import argparse
from pathlib import Path
import sys
import hashlib
import carla

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"integration"))
from india_road import junction_xml,LAYOUT_ARMS
from india_dataset_maps import validate_layout
from india_streetscape import paving_layout,footprint_samples
from prepare_simulation_dataset import DEFAULT,save

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output",type=Path,default=DEFAULT)
output=parser.parse_args().output.resolve()

results=[]
for layout,active in LAYOUT_ARMS.items():
    xml=junction_xml(layout)
    checked=validate_layout(xml,layout)
    road_map=carla.Map(layout,xml)
    count=0
    for tile in paving_layout(active):
        for x,y in footprint_samples(tile):
            assert road_map.get_waypoint(carla.Location(x=x,y=y,z=0),False,carla.LaneType.Driving) is None,(layout,tile,x,y)
            count+=1
    checked.update(footpath_samples_checked=count,map_sha256=hashlib.sha256(xml.encode()).hexdigest())
    results.append(checked)
assert len({r["map_sha256"] for r in results})==4
report={"passed":True,"layouts":results,"distinct_map_hashes":4}
save(output/"map_preflight.json",report)
print(json.dumps(report,indent=2))
