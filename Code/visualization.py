import matplotlib.pyplot as plt
import os
import numpy as np

class Visualization:
    def __init__(self, path, dpi):
            self._path = path
            self._dpi = dpi


    def save_data_and_plot(self, data, filename, xlabel, ylabel):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt
        """
        min_val = min(data)
        max_val = max(data)
        n = 10
        moving_average = [0]*9 + [np.mean(data[i:i+n]) for i in range(len(data) - n + 1)]
        
        plt.rcParams.update({'font.size': 24})  # set bigger font size
        
        plt.plot(data, label='data')
        plt.plot(moving_average, label='moving average')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.show()
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)

        with open(os.path.join(self._path, 'plot_'+filename + '_data.txt'), "w") as file:
            for value in data:
                    file.write("%s\n" % value)
    
