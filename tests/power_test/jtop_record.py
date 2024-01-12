from jtop import jtop
import pandas as pd
n = 100
count = 0
result = []
device_name = 'xavier'
operator_name = 'spare'
version = 'spare'
with jtop() as jetson:
    # jetson.ok() will provide the proper update frequency
    while jetson.ok():
        if count >= n:
            break
        # Read tegra stats
        power = jetson.stats['Power TOT']
        print(power)
        power = int(power)
        result.append(power)
        count += 1

data = {'power(mW)': result}
df = pd.DataFrame(data)

filename = f"power_measure/power_{device_name}_{operator_name}_{version}.csv"
df.to_csv(filename, index=False)
