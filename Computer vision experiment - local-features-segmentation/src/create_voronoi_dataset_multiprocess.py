import concurrent
import sys;

sys.path.append('../')
import numpy as np
import tensorflow as tf
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from src.utils.graphs import get_keypoints, relative_neighborhood, random_graph
from src.utils.keypoints_filter import mean_distance_filter, model_filltering, load_filter_model
from scipy.spatial.distance import cdist
from src.utils.visualize import plot_voronoi
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
from shapely.geometry import Polygon, Point
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2 as cv

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 120
JSON_EXT = ".json"
JPG_EXT = ".jpg"
color = {
    1: "brown",
    2: "blue",
    3: "green",
    4: "red"
}
WRONG_CLASS = 3
JSON_EXT = ".json"
JPG_EXT = ".jpg"
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 120
FILTER_RADIUS = 8


def filter_polygons(polygon, img_shape):
    x = img_shape[0]
    y = img_shape[1]
    (miny, minx, maxy, maxx) = [int(item) for item in list(polygon.bounds)]
    if minx <= -1 or maxx > x:
        return False
    if miny <= -1 or maxy > y:
        return False
    return True


def get_label(polygon, points_df):
    x = points_df.x
    y = points_df.y
    labels = points_df.label_id
    possible_labels = []
    for idx, xs in enumerate(x):
        p = Point(x[idx], y[idx])
        if polygon.contains(p):
            possible_labels.append(labels[idx])

    unique_labels, unique_counts = np.unique(possible_labels, return_counts=True)
    if len(unique_labels) >= 1:
        # print("possible_labels for region",unique_labels,unique_counts)
        idx = np.argmax(unique_counts)
        return unique_labels[idx]
    elif len(unique_labels) >= 0:
        return WRONG_CLASS


def preprocess_keypoints(keys, radius=FILTER_RADIUS):
    keys = mean_distance_filter(keys, plot_filter=False, min_distance=radius)
    return keys


def walls(max_x, max_y):
    STEP = 100
    points = []
    for i in range(0, max_x, STEP):
        points.extend([[i, max_y], [i, 0]])
    for i in range(0, max_y, STEP):
        points.extend([[max_x, i], [0, i]])
    return points


def keypoints_to_list(keys):
    keys_list = []
    [keys_list.append(k.pt) for k in keys if k.pt not in keys_list]
    return keys_list


def get_regions(image_name, plot=False, FILTER_MODEL=load_filter_model()):
    json_path = image_name + JSON_EXT

    json_df = pd.read_json(json_path)
    if json_df.empty:
        return (False, False)

    cell_points = list(zip(json_df.x, json_df.y))
    radius = cdist(cell_points, cell_points)
    radius = radius[radius > 0].min() * 0.5
    # print(radius)
    img_path = image_name + JPG_EXT
    img_arr = plt.imread(img_path)
    labels = json_df.label_id

    sift = sift = cv.SIFT_create(
        nOctaveLayers=40,
        contrastThreshold=0.02,
        edgeThreshold=30,
        sigma=1.3)
    # filter sifts with dl model
    keys, desc = sift.detectAndCompute(img_arr, None)
    filter_model = FILTER_MODEL
    keys = model_filltering(filter_model, keys, desc)
    max_x = img_arr.shape[0] - 1
    max_y = img_arr.shape[0] - 1
    wall = walls(max_x, max_y)
    keys = keypoints_to_list(keys)
    keys = preprocess_keypoints(keys, radius)
    keys.extend(wall)
    vor = Voronoi(keys)
    polygons = []

    tmp_df = json_df.copy()
    labels_list = []
    if plot:
        plot_voronoi(vor, img_arr, points_size=5, show_points=True, show_verticles=False, line_width=0.3)
    for idx, region in enumerate(vor.regions):
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            x = Polygon(polygon)
            if x.is_empty:
                continue

            if filter_polygons(x, img_arr.shape):
                label = get_label(x, tmp_df)

                polygons.append(x)
                # print(label)
                labels_list.append(label)
                if plot:
                    plt.fill(*zip(*polygon), color[label], alpha=0.5)
        else:
            # print(region,[vor.vertices[v] for v in region if v!=-1])
            pass
    if plot:
        plot_labels(json_df)
        plt.show()
    return polygons, labels_list


def plot_labels(df):
    for idx, row in df.iterrows():
        x = row[0];
        y = row[1];
        label = row[2]
        plt.plot(x, y, marker='^', color=color[label])


def plot(image_name):
    json_path = image_name + JSON_EXT

    json_df = pd.read_json(json_path)
    cell_points = list(zip(json_df.x, json_df.y))
    vor = Voronoi(cell_points)
    img_path = image_name + JPG_EXT
    img_arr = plt.imread(img_path)
    plot_voronoi(vor, img_arr, show_points=True, show_verticles=False, line_width=0.3)
    # plt.gca().invert_xaxis()
    # plt.gca().invert_yaxis()


def plot_arr(img_arr):
    fig, ax = plt.subplots()
    ax.imshow(img_arr)
    # ax.set_ylim([0, 180])
    # ax.set_xlim([0, 180])
    # plt.gca().invert_yaxis()
    # plt.gca().invert_xaxis()
    return fig


def get_pixels(image_name, polygon, plot=False):
    img_path = image_name + JPG_EXT
    img_arr = plt.imread(img_path)

    # plot_arr(img_arr)
    # print(img_arr.shape)

    (miny, minx, maxy, maxx) = [int(item) for item in list(polygon.bounds)]
    # print(polygon.bounds)
    shape = img_arr.shape
    x_range = range(minx, maxx)
    y_range = range(miny, maxy)
    if plot:
        test = np.copy(img_arr)
    mask = np.copy(img_arr[minx:maxx, miny:maxy])
    for x in x_range:
        for y in y_range:
            p = Point(y, x)
            if not polygon.contains(p):
                if plot:
                    test[x, y] = test[x, y] * [255, 0, 0]
                mask[x - minx, y - miny] = mask[x - minx, y - miny] * [0, 0, 0]
    #
    if plot:
        plot_arr(test)
        plot_arr(mask)
    return mask


def process_img(params):
    filename = params[0]
    filepath = params[1]
    DIST = params[2]
    print(filename, filepath)
    try:
        df = pd.DataFrame(columns=['filename', 'label'])
        polygons, labels = get_regions(filepath, False)
        if polygons is False:
            return df
        it = 0
        for idx, polygon in tqdm(enumerate(polygons)):
            try:
                arr = get_pixels(filepath, polygon, False)
                im = Image.fromarray(arr)
                f = filename + "_" + str(it) + ".jpg"
                dist = os.path.join(DIST, f)
                it += 1
                l = labels[idx]
                df_tmp = pd.DataFrame({
                    'filename': [f],
                    'label': [l]
                })
                df = df.append(df_tmp)
                im.save(dist)
            except Exception as e:
                print("error with polygon", filename, e)
        # plt.show()
        # break

        return df
    except Exception as e:
        print("Error with file: ", filename, " : ", e)


def create_set_multiprocess(path, distpath):
    if not os.path.exists(distpath):
        os.mkdir(distpath)
    DIST = distpath
    print('Working at: ', path)
    filelist = []
    filter_model = load_filter_model()
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            filename = file.split('.')[0]
            ext = file.split('.')[1]
            if ext == 'jpg':
                filepath = os.path.join(root, filename)
                if os.path.exists(filepath + '.json') and os.path.getsize(filepath + '.json') > 1:
                    filelist.append((filename, filepath, DIST))
    df = pd.DataFrame(columns=['filename', 'label'])
    results_dfs = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        results_exe = executor.map(process_img, filelist)
        for res in tqdm(results_exe, total=len(filelist), desc=f"progress"):
            results_dfs.append(res)
    for res_df in results_dfs:
        df = df.append(res_df, ignore_index=True)
    df.to_csv(os.path.join(DIST,"label.csv"))

if __name__ == "__main__":
    train_path = r"../data/Ki67/SHIDC-B-Ki-67/Test/"
    #train_path = r"../test_images/"
    data_type = "Testing_solution"
    DIST = "../dl/model_filtered_Test"
    prefix = str(FILTER_RADIUS) + "_" + data_type

    create_set_multiprocess(train_path, DIST)
