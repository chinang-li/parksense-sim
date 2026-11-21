import pandas as pd
from experiments import runs, save, BASE
cfg = dict(BASE, areas_x=4, areas_y=3, rows_per_area=7, spaces_per_row=12)
rows=[]
for p,k in ((0.0,5),(0.0,10),(30.0,10)):
    rows.append(runs(dict(cfg, info_period_s=p, guide_topk=k), ["GUIDE"], 30, spaces=1008, period=p, topk=k))
s=save(pd.concat(rows),"S12b_scale_guide_variants",["method","spaces","period","topk"])
print(s[['time_to_park_s_mean','conflicts_per_100','area_imbalance_sd','hot_area_h','walk_m_mean','drive_m_per_veh']].round(2).to_string())
