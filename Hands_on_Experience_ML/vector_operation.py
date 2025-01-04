import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

plt.quiver(0,0,1,1)

plt.xlim(-8,8)
plt.ylim(-8,8)
plt.quiver(0,0,4,5, scale=1, scale_units='xy', angles='xy', color='b')

plt.show()

plt.xlim(-8,8)
plt.ylim(-8,8)
plt.quiver(0,0,4,5, scale=1, scale_units='xy', angles='xy', color='b')
plt.quiver(0,0,6,7, scale=1, scale_units='xy', angles='xy', color='r')

plt.show()

#addition of vectors
vector1 = np.array([0,0,2,3])
vector2 = np.array([0,0,3,2])
vector3 = vector1 + vector2
print(vector3)

plt.xlim(-8,8)  
plt.ylim(-8,8)
plt.quiver(0,0,2,3, scale=1, scale_units='xy', angles='xy', color='b')  
plt.quiver(0,0,3,-2, scale=1, scale_units='xy', angles='xy', color='r')  
plt.quiver(0,0,5,1, scale=1, scale_units='xy', angles='xy', color='g')
plt.show()  