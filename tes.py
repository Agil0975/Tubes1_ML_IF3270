import numpy as np

history = np.zeros((10, 2))
history[0, 0] = 1.0
history[0, 1] = 2.0
history[9, 0] = 3.0
history[9, 1] = 4.0

print(history)
print(history[:5])