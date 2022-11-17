import pandas as pd
import json

f = open("TVs-all-merged.json")
data = json.load(f)
f.close()
print(data["47LW5600"])
for i in data:  # i is model key
    for j in data[i]:  # j is dict that represents one site
        del j["url"]
df = pd.DataFrame()

for i in data:  # i is model key
    for j in data[i]:  # j is dict that represents one site
        a = pd.DataFrame.to_dict(j)
