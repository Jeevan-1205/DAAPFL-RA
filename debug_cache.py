import os
import numpy as np

cache = sorted(os.listdir("outputs/cache/tiles"))[0]

path = os.path.join("outputs/cache/tiles", cache)

z = np.load(path)

pre = z["pre"]
post = z["post"]

print("Raw cache difference:", np.abs(pre - post).mean())
print("Identical:", np.all(pre == post))