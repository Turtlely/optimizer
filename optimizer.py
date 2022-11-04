# Performs a linear regression, maps out cost as a function of m and b

# Generate a set of random points following a linear trend
# Function to be mapped

import numpy as np
import matplotlib.pyplot as plt

# Noising the test signal
noise = np.random.normal(0,1,100)

def FUNCT(x):
    b = 1
    m = 2
    return m * x + b

x = np.linspace(0,10,100)
y = FUNCT(x)
y_noise = y + noise

# Begin fitting

#cost function
def cost(f,m,b):
    # Calculate error squared
    y_pred = m * x + b
    y = f(x)

    error = np.square((y-y_pred))

    return np.average(error)


# Parameters to optimize


X=np.linspace(105,10,100)
Y=np.linspace(-10,10,100)

M, B = np.meshgrid(X,Y)
# M is x axis
# B is y axis

Z = [ [0 for _ in range(100)] for _ in range(100)]

for xax in range(100):
    for yay in range(100):
        Z[xax][yay] = cost(FUNCT,X[xax],Y[yay])

Z = np.array(Z)

fig = plt.figure()
fig2 = plt.figure()
#fig3 = plt.figure()

ax = fig.gca(projection = '3d')
ax2 = fig2.add_subplot()
#ax3 = fig3.add_subplot()

surface = ax.plot_surface(M, B, Z,cmap='Blues_r',antialiased=False)
ax.set_xlabel("Slope")
ax.set_ylabel("Intercept")
ax.set_zlabel("Cost")

ax2.plot(x,y,c='red')
ax2.scatter(x,y_noise)

plt.show()
