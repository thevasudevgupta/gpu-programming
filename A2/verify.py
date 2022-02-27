import numpy as np

C = np.array([[1, 9, 6], [-6, 7, 2], [2, 4, -3]])
D = np.array([[10, 0, 5], [1, 3, -3]])
print(C@D.T)

addn = np.array([[8, 1, 1], [4, 0, 4]])
print(addn@(C@D.T))
