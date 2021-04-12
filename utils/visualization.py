import matplotlib.pyplot as plt
import numpy as np

def draw_line_graph(x, y1, y2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y1, '-', label='train loss')
    ax2 = ax.twinx()
    ax2.plot(x, y2, '-r', label='test metrics')
    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

    ax.set_xlabel("Epoch id")
    ax.set_ylabel(r"train loss")
    ax2.set_ylabel(r"test metrics")
    plt.show()


def draw_line_graph_by_list(input_list1,input_list2):
    n = len(input_list1)
    draw_line_graph(range(1, n+1), input_list1, input_list2)

def draw_feature_graph(feature_signal):
    n = len(feature_signal)
    plt.plot(range(1,n+1), feature_signal, '-')
    plt.show()


def draw_feature_map(num_channel,ouput_feature):
    fig = plt.figure()
    for i in range(num_channel):
        ax = fig.add_subplot(16, 16, i+1, xticks = [], yticks = [])
        ax.plot(draw_feature_graph(ouput_feature[:][i]))
    plt.show()
