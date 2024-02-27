from jtop import jtop
import pandas as pd
import sys
n = 30
count = 0
result = []
device_name = 'xavier'
operator_name = 'human'
version = 'tinyyolov3'
interval = sys.argv[1]
with jtop() as jetson:
    # jetson.ok() will provide the proper update frequency
    while jetson.ok():
        if count >= n:
            break
        if count <= 10:
            count += 1
        else:
            # Read tegra stats
            power = jetson.stats['Power TOT']
            #print(power)
            power = int(power)
            result.append(power)
            count += 1

data = {'power(mW)': result}
df = pd.DataFrame(data)

filename = f"power_{device_name}_{operator_name}_{version}_{interval}.csv"
df.to_csv(filename, index=False)

