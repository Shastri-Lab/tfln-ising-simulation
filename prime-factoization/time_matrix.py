import matplotlib.pyplot as plt

nbits = [4, 8, 16, 32]
time = [0.00018596649169921875, 0.0024080276489257812, 0.18893218040466309 ,33.32747936248779]

plt.plot(nbits,time)
plt.show()