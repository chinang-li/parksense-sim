"""Check: the simulator's scoring reproduces the real web dashboard (prototype sheet ParkingAllList.csv)."""
import csv
import numpy as np
from config import BASE
from model import Sim, RESERVED

rows = list(csv.DictReader(open("ParkingAllList.csv", encoding="utf-8-sig")))
names = sorted({r["area"] for r in rows})
L = dict(n=len(rows), na=len(names),
         AREA=np.array([names.index(r["area"]) for r in rows]),
         AE=np.array([float(r["area_score"]) for r in rows]), SE=np.array([float(r["space_score"]) for r in rows]),
         POS=np.array([float(r["position_score"]) for r in rows]), OK=np.array([r["in_service"] == "1" for r in rows]))
label = [f'{r["area"]}-{r["space"].zfill(2)}' for r in rows]
sim = Sim(dict(BASE), "PARKSENSE", L, [])

def show(tag):
    sc = sim.scores("reservation")
    return {label[i]: round(float(sc[i]), 2) for i in range(len(rows)) if L["OK"][i] and sim.status[i] == 0}

before = show("before")
sim.set_status(label.index("A-01"), RESERVED, 1)
after = show("after")
print("before:", before)
print("after A-01:", after)
assert before["A-01"] == 0.75 and before["A-03"] == 0.70 and before["A-04"] == 0.60
assert after["A-03"] == 0.63 and after["A-04"] == 0.53
print("OK - matches the dashboard (A-01 0.75, A-03 0.70->0.63, A-04 0.60->0.53)")
