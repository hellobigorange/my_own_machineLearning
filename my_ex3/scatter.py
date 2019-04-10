import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))


# 动画变化保持y变化，x不变
def animate(i):
    line.set_ydata(np.sin(x+i/10))
    return line,


# 最开始的一帧动画
def init():
    line.set_ydata(np.sin(x))
    return line,


# frames帧，时间点，循环周期,init_func最开始的动画,blit更新所有点or变化的点
ani = animation.FuncAnimation(fig=fig, func=animate, frames=100, init_func=init, interval=20, blit=False)
plt.show()
print('RUN off')


