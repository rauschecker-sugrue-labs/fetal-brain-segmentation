import os
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import OffsetImage,AnnotationBbox

try:
    iroot = Path('Icons')
    figures_dir = Path('Figures') / os.environ["USER"] 
    figures_dir.mkdir(exist_ok=True)
except:
    pass

def multi_plot(plot_type, dfs, metric=None, save=None, right_axis=True, icons=False, rotate_labels=False, colors='#4472C4', **plt_kwargs):
    """ Plots several graphs on the same y axis
    Example
    -------
        multi_plot(sns.boxplot, dfs=[df1,df2], save='fig_name', colors=['#4472C4','#70AD47'], **{'width': 0.6});
    """
    n = len(dfs)
    if colors is None:
        colors = ['#4472C4'] * n
    elif type(colors) is str:
        colors = [colors] * n
    elif len(colors) != n:
        print('Wrong colors argument: does not match len(dfs).\nUsing default.')
        colors = ['#4472C4'] * n
    cols_length = [len(df.columns) for df in dfs]
    total_col_length = np.sum(cols_length)

    f = plt.figure(constrained_layout=True, figsize=(total_col_length+1,5))
    gs = f.add_gridspec(1,total_col_length, wspace=0)

    fs = []
    start = 0
    for inum, df in enumerate(dfs):
        end = start + cols_length[inum]
        fi = f.add_subplot(gs[0,start:end])
        start = end
        plot_type(data=df, color=colors[inum], ax=fi, **plt_kwargs)  # green: #70AD47
        # fi.yaxis.set_visible(False)
        if inum >0: fi.set_yticks([])
        fs.append(fi)
    
    # Set left axis label
    if metric and metric != 'd':
        fs[0].set_ylabel(metric, fontsize=12)
    else:
        fs[0].set_ylabel('Dice score', fontsize=12)

    # Set right axis label
    if right_axis:
        fs[-1].yaxis.tick_right()
        fs[-1].yaxis.set_label_position("right")

    for ax in f.axes:
        plt.sca(ax)
        if rotate_labels: plt.xticks(rotation=40, horizontalalignment='right');
        plt.ylim(0,1)
        plt.tick_params(axis='y', which='both', right=False)

    if icons:
        for inum, df in enumerate(dfs):
            show_icons(df, fs[inum])

    sns.despine(left=True, right=True)
    f.tight_layout()

    if save:
        savefig(f, save)
    
    return [f, fs]


def get_icon(name):

    if '.' in name: # takes care of the 'exp.metric' (e.g. 102P.FDR) #TODO changed that to space... maybe better way than just one character?
        name = name.split('.')[0]
    if 'block' in name or name == 'all':
        icon_path = iroot / 'ft.png'
    else:
        icon_path = iroot / (name+'.png')
    try:
        im = plt.imread(icon_path)
    except:
        return None
    return im

def offset_image(coord, name, ax):
    img = get_icon(name)
    if img is None:
        return
    im = OffsetImage(img, zoom=0.04)
    im.image.axes = ax
    ab = AnnotationBbox(im, (coord, 0), xybox=(0., -35.), frameon=False, xycoords='data', boxcoords="offset points", pad=0)
    ax.add_artist(ab)

def show_icons(df, ax):
    for i, c in enumerate(df.columns):
        offset_image(i, c, ax)

def savefig(fig, name, transparent=False):
    if not isinstance(fig, plt.Figure):
        fig.get_figure().savefig(figures_dir/(name+'.png'), dpi=300, transparent=True, bbox_inches="tight")
    else:
        fig.savefig(figures_dir/(name+'.png'), dpi=300, transparent=transparent, bbox_inches="tight")


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def slices_viewer(X):
    """ Scroll through 2D image slices of a 3D array.
        ex:  slices_viewer(np.random.rand(20, 20, 40))
        NB:  only works in notebook environment. Run `%matplotlib widget` first.
    """
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, X)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    # plt.show()
    return fig