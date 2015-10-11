#!/usr/bin/env python
# coding=utf-8

import numpy as np 
from matplotlib import pyplot as plt 
from matplotlib import animation 
# First set up the figure, the axis, and the plot element we want to animate 
fig = plt.figure() 
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2)) 
line, = ax.plot([], [], lw=2) 
print line
# initialization function: plot the background of each frame 
def init(): 
  line.set_data([], []) 
  return line, 
# animation function. This is called sequentially 
# note: i is framenumber 
def animate(i): 
  x = np.linspace(0, 2, 1000) 
  y = np.sin(2 * np.pi * (x - 0.01 * i)) 
  line.set_data(x, y) 
  return line, 
# call the animator. blit=True means only re-draw the parts that have changed. 
anim = animation.FuncAnimation(fig, animate, init_func=init, 
                frames=200, interval=20, blit=True) 
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264']) 
plt.show() 
