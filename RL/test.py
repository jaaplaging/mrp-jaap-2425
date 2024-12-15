import numpy as np

array = np.full((6, 5), True)

array[0,:] = [False, False, False, False, False]

print(array)