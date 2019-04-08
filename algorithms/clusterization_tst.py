import numpy as np
import numpy.random as rnd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

data = rnd.randint(0, 100, (30, 30), dtype=np.int32)
data_list = list()
for x in range(30):
    for y in range(30):
        for a in range(data[x][y]):
            data_list.append([x, y])
model = DBSCAN()
clusters = model.fit_predict(data_list)
print(clusters)

plt.imshow(data)
plt.show()
