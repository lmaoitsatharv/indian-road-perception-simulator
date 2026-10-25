"""Independent saved-PNG recount for completed adaptation validation exports."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pygame


def channel(path):
    return pygame.surfarray.array3d(pygame.image.load(str(path)))[:,:,0].T


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    output=args.output.resolve()
    manifest=json.loads((output/"manifest.json").read_text(encoding="utf-8"))
    state=json.loads((output/"run_status.json").read_text(encoding="utf-8"))
    assert state["phase"]=="completed" and state["protected_files_unchanged"]
    root=Path(__file__).resolve().parents[1]
    for name,expected in manifest["protected_files"].items():
        with (root/name).open("rb") as stream:
            assert hashlib.file_digest(stream,"sha256").hexdigest()==expected
    def local(path):
        value=Path(path)
        if path.startswith("/mnt/e/"):
            value=Path("E:/"+path[len("/mnt/e/"):])
        return value
    selection=json.loads((output/"selection.json").read_text(encoding="utf-8"))
    datasets={label:manifest["real_validation"] if label.endswith("_real") else manifest["development"]
              for label in ("baseline_real","epoch_01_real","epoch_02_real","baseline_development","epoch_01_development","epoch_02_development")}
    if selection["selected_epoch"]:
        heldout=json.loads((output/"data/heldout_after_selection/manifest.json").read_text(encoding="utf-8"))["records"]
        datasets.update(baseline_heldout=heldout,candidate_heldout=heldout)
    checked={}
    for label,rows in datasets.items():
        folder=output/"evaluations"/label
        reported=json.loads((folder/"metrics.json").read_text(encoding="utf-8"))
        by_name={row["name"]:row for row in reported["rows"]}
        totals=dict(tp=0,fp=0,fn=0,tn=0,ignored=0,total=0)
        for row in rows:
            truth=channel(local(row["mask"]))
            mask=channel(folder/"masks_01"/(row["name"]+".png"))
            assert set(np.unique(mask)) <= {0,1} and set(np.unique(truth)) <= {0,1,255}
            good=truth!=255;t=truth==1;p=mask==1
            actual={"tp":int((good&t&p).sum()),"fp":int((good&~t&p).sum()),"fn":int((good&t&~p).sum()),
                    "tn":int((good&~t&~p).sum()),"ignored":int((~good).sum()),"total":int(truth.size)}
            for key in totals:
                assert actual[key]==by_name[row["name"]]["metrics"][key]
                totals[key]+=actual[key]
        for key in totals:assert totals[key]==reported["metrics"][key]
        checked[label]={"images":len(rows),"counts":totals}
    if not selection["selected_epoch"]:
        assert not (output/"data/heldout_after_selection").exists()
    result={"passed":True,"independently_recounted":checked,
            "protected_files_unchanged":True,"withheld_opened_only_for_qualified_candidate":True,
            "method":"Direct saved-PNG 0/1/255 confusion recount, without importing training or inference code"}
    (output/"independent_recount.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"passed":True,"datasets":{key:value["images"] for key,value in checked.items()}},indent=2))


if __name__=="__main__":
    main()
