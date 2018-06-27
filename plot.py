import numpy as np
import matplotlib.pyplot as plt


class Plotter(object):
    fig_number = 0

    def __init__(self, title=None, semilogy=False, ylim_max=None):
        plt.ion()  # You probably won't need this if you're embedding things in a tkinter plot...
        self.num = Plotter.fig_number
        Plotter.fig_number += 1
        #self.fig = plt.figure(self.num)
        self.data = {}
        self.title = title
        self.semilogy = semilogy
        self.ylim_max = ylim_max

    def add_value(self, x, y, label):
        if label in self.data:
            if x is None:
                x = len(self.data[label]['x'])
            self.data[label]['x'].append(x)
            self.data[label]['y'].append(y)
        else:
            if x is None:
                x = 0
            self.data[label] = {
                'x': [x],
                'y': [y]
            }

    def draw(self):
        self.fig = plt.figure(self.num)
        plt.cla()
        plt.grid()
        if self.title:
            plt.title(self.title)
        for label in self.data:
            if self.semilogy:
                plt.semilogy(self.data[label]['x'], self.data[label]['y'], label=label)
            else:
                plt.plot(self.data[label]['x'], self.data[label]['y'], label=label)
        #plt.xlabel("Time [s]")
        #plt.ylabel()
        if self.ylim_max:
            plt.ylim(ymax=self.ylim_max)
        plt.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
