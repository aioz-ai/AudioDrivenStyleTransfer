import json 
from pathlib import Path 
import pandas as pd 
import numpy as np 
inputfolder = Path("standard_lmd")
data_dict = {}
for fn in sorted(list(inputfolder.glob("*.json"))):
    method = fn.name
    try:
        with open(fn) as f:
            # print(fn)
            data = json.load(f)
        if "col" not in data_dict.keys():
            data_dict["col"] = [method]
        else:
            data_dict["col"].append(method)
        for k,v in data.items():
            if k == "file":
                continue 
            if k not in data_dict.keys():
                data_dict[k] = [v[-1]]
            else:
                data_dict[k].append(v[-1])
    except Exception as e:
        print("Error: ", e)
df = pd.DataFrame(data=data_dict)
df.to_csv("abcxx.csv")
smds = []
slvs = [] 
slds = []
for k,v in df.iterrows():
    col,smd,slv,sld,length,stride = v
    smds.append(smd)
    slvs.append(slv)
    slds.append(sld)

print(f"SMD: {np.mean(smds)}, SLV: {np.mean(slvs)}, SLD: {np.mean(slds)}")