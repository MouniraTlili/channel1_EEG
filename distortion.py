import numpy as np
import matplotlib.pyplot as plt

x = [0,5,10,30,40,50,70,80,90]
# Make an array of y values for each x value
y = [0.38, 0.52, 0.65, 3.48, 7.48, 12.67, 25.79, 35.90, 47.26]
z = [0.59, 0.79, 0.98, 7.36, 15.6, 21.54, 37.47, 42.62, 49.38]
plt.grid(True)
plt.xlabel('Compression %')
plt.ylabel('Distortion %')
# plt.axis([0, 100, 0, 100])
plt.title('Plot of distortion % vs. Compression %')
plt.plot(x,z, 'bo')
plt.plot(x,z, 'r')

for xy in zip(x, z):                                       # <--
    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data',ha='right', va='bottom',color='gray') # <--
    # plt.annotate(
    #     xy=(x, y), xytext=(-20, 20),
    #     textcoords='offset points', ha='right', va='bottom',),
    #     arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.show()