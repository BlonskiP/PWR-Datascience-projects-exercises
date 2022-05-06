# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import sys; sys.path.append('../')
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.utils.visualize import plot_voronoi
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point
from PIL import Image
import numpy as np


# %%
#train_set = r"../data/Ki67/SHIDC-B-Ki-67/Train/p1_0308_3"
train_set = r"../test_images/dots"
JSON_EXT = ".json"
JPG_EXT = ".jpg"
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 120


# %%
def get_regions(image_name):
    json_path = image_name+JSON_EXT

    json_df = pd.read_json(json_path)

    cell_points = list(zip(json_df.x,json_df.y))
    vor = Voronoi(cell_points)
    polygons = []
    for idx ,region in enumerate(vor.regions):
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            x = Polygon(polygon)
            if x.is_empty:
                continue
            
            plt.fill(*zip(*polygon))
    img_path = image_name+JPG_EXT
    img_arr = plt.imread(img_path)
    plot_voronoi(vor,img_arr,show_points=True,show_verticles=True,line_width=0.3)
    polygons.append(Polygon(polygon))
    plt.show()
    return polygons


# %%

def plot_arr(img_arr):
    fig, ax = plt.subplots()
    ax.imshow(img_arr)
    ax.set_ylim([0, 127])
    ax.set_xlim([0, 110])
   # plt.gca().invert_yaxis()
    return fig


# %%



# %%
def get_pixels(image_name,polygon):
    img_path = image_name+JPG_EXT
    img_arr = plt.imread(img_path)
    (minx, miny, maxx, maxy)=[int(item) for item in list(polygon.bounds)]
    x_range = range(minx,maxx-1)
    y_range = range(miny,maxy-1)
    shape = img_arr.shape
    print('shape',shape)
    print('min/max',(minx, miny, maxx, maxy))
    region_mask = np.zeros((maxx-minx,maxy-miny,3))
    region_mask = np.zeros((maxx,maxy,3))
    
    x_inside=minx-1; y_inside=miny-1
    for x in x_range:
        x_inside +=1
        if x >= shape[0] or x <= -1:
            continue
        for y in y_range:
            y_inside +=1
            if y >= shape[1] or y <=-1:
                continue
            p = Point(x,y)
            if polygon.contains(p):
                #pixels.append(img_arr[x][y])
                #print('test',x,y,x_inside,y_inside)
                region_mask[x_inside][y_inside]=img_arr[x][y]
        y_inside= miny-1
        
    
    plot_arr(region_mask)
    print('inside',x_inside,y_inside)
    print('minx-maxx',maxx-minx)
   # print(region_mask[60][60])


# %%
polygons = get_regions(train_set)


# %%
polygons


# %%
get_pixels(train_set,polygons[0])


# %%



