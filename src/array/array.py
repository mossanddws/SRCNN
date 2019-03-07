import numpy as np

a = np.arange(0, 6)
b = np.arange(6, 12)
c = np.arange(12, 18)
d = np.arange(18, 24)
e = np.asarray([a, b, c, d])

print(e)

print(e[0:3, 0:3])
print(e[0:3, 1:4])
print(e[0:3, 2:5])

f = np.zeros((2, 4, 3, 3))
print(f)
print('------------------------------------------------------------')
for i in range(0, 2):
    for j in range(0, 4):
        f[i, j] = e[i:i + 3, j:j + 3]

print(f)
