#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 11:13:59 2020

@author: tjards
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
fig.set_tight_layout(True)



# Plot a scatter that persists (isn't redrawn) and the initial line.
x = np.arange(0, 20, 0.1)
ax.scatter(x, x + np.random.normal(0, 3.0, len(x)))
line, = ax.plot(x, x - 5, 'r-', linewidth=2)

def update(i):
    label = 'timestep {0}'.format(i)
    print(label)
    # Update the line and the axes (with a new xlabel). Return a tuple of
    # "artists" that have to be redrawn for this frame.
    line.set_ydata(x - 5 + i)
    ax.set_xlabel(label)
    return line, ax


anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=200)
anim.save('test.gif', writer='imagemagick') #my add





# if len(sys.argv) > 1 and sys.argv[1] == 'save':
#     anim.save('line.gif', dpi=80, writer='imagemagick')
# else:
#     # plt.show() will just loop the animation forever.
#     plt.show()


# if __name__ == '__main__':
#     # FuncAnimation will call the 'update' function for each frame; here
#     # animating over 10 frames, with an interval of 200ms between frames.
#     anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=200)
#     if len(sys.argv) > 1 and sys.argv[1] == 'save':
#         anim.save('line.gif', dpi=80, writer='imagemagick')
#     else:
#         # plt.show() will just loop the animation forever.
#         plt.show()