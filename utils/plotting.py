import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List,Tuple
import math


def plot_line_graphs_overlayed(x_list:List[np.ndarray],
                                  y_list:List[np.ndarray],
                                  x_axe_name:str, 
                                  y_axe_name:str,
                                  legends:List[str],
                                  title:str,
                                  legend_position:str='upper right',
                                  graph_saving_path:str = None,
                                  figsize:Tuple[int] = (10,10)):
    
    """Plot multiple graphs together .
    Args:
        x_list (List[np.ndarray]): input data x axis list.
        y_list (List[np.ndarray]): input data y axis list.
        x_axe_name (str): x axe name.
        y_axe_name (str): y axe name.
        legends (List[str]): legends list.
        title (str): graph title.
        legend_position (str): legend position on graph.
        figsize (Tuple[int]): tuple with figsize dimentions.
    Returns:
        (matplotlib.Axes): the axes object.
    """
    if not (len(x_list) == len(y_list) == len(legends)):
        raise ValueError(f"Legends (with length {len(legends)}) must have the same lenght as x_list (with length {len(x_list)}) and y_list (with length {len(y_list)})")
        
    cmap = sns.color_palette("Spectral_r", 256)
    color_map = list(cmap)
    color_options = [color_map[20],color_map[-20], color_map[40],color_map[-40],color_map[255],color_map[-90]]
    n_repetitions = math.ceil(len(color_options)/len(x_list))
    color_repetitions = n_repetitions*color_options
    colors = color_repetitions[:len(x_list)]

    line_styles_options = ['-', '--', '-.', ':']
    line_styles = sum([[line_style]*len(color_options) for line_style in line_styles_options],[])[:len(x_list)]

    marker_styles = ['']*len(line_styles)

    fig, ax = plt.subplots()
    plot_list = []

    for x,y,line_style,marker_style,color,legend in zip(x_list,y_list,line_styles,marker_styles,colors,legends):
      if legend ==  "None_Marker":
        plot_list.append(ax.plot(x, y, color=color, linestyle=line_style,marker=marker_style))
      else:
        plot_list.append(ax.plot(x, y, color=color, linestyle=line_style,marker=marker_style,label=legend))
      

    if legends !=  ["None_Marker"]:
      ax.legend(loc=legend_position)
    ax.set_xlabel(x_axe_name,fontsize = 10)
    ax.set_ylabel(y_axe_name,fontsize = 10)
    ax.set_title(title,fontsize = 20)
    ax.figure.set_size_inches(figsize[0], figsize[1])
    
    plt.savefig(graph_saving_path)
    plt.show()
    # plt.draw()
    fig.tight_layout()


    plt.clf()

    return ax
