import concurrent.futures
import math
import os
import pickle
import sys, os
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
from itertools import combinations
from PIL import Image
from tqdm import tqdm
from math import sin, cos, pi, log
from matplotlib.patches import Ellipse
from shapely.geometry import Polygon
import statistics
from statistics import NormalDist, mean, median

MEAN_ELLIPSE_PARAMS_BLUE = {
    'radius_x': 18,
    'radius_y': 20,
}
MEAN_ELLIPSE_PARAMS_BROWN = {
    'radius_x': 33,
    'radius_y': 35
}
data_path = "../data/Ki67/SHIDC-B-Ki-67/Train"
train_set = r"../data/Ki67/SHIDC-B-Ki-67/Train"


def check_ellipse_rotated(ellipse, x, y):
    radius_x = ellipse['radius_x']
    radius_y = ellipse['radius_y']
    center_x = ellipse['center_x']
    center_y = ellipse['center_y']
    angle = ellipse['theta']

    ellipse_eq_1 = (((x - center_x) * cos(angle) + (y - center_y) * sin(angle)) ** 2) / (radius_x) ** 2
    ellipse_eq_2 = (((x - center_x) * sin(angle) - (y - center_y) * cos(angle)) ** 2) / (radius_y) ** 2
    if ellipse_eq_1 + ellipse_eq_2 <= 1:
        return True
    else:
        return False


def get_ellipse(center_x, center_y, theta, radius_sum, img, radius_x, radius_y):
    start_x = center_x - radius_sum
    start_y = center_y - radius_sum
    if start_x < 0:
        start_x = 0
    if start_y < 0:
        start_y = 0
    end_x = center_x + radius_sum
    end_y = center_y + radius_sum
    if end_x > img.width:
        end_x = img.width
    if end_y > img.height:
        end_y = img.height
    pix_color_inside = []
    x_range = range(start_x, end_x)
    y_range = range(start_y, end_y)
    ellipse = {
        'center_x': center_x,
        'center_y': center_y,
        'theta': theta,
        'colors': None,
        'radius_x': radius_x,
        'radius_y': radius_y,
        'label': 1
    }
    for x in x_range:
        for y in y_range:
            if check_ellipse_rotated(ellipse, x, y):
                pix_color_inside.append(img.getpixel((x, y)))
    ellipse['colors'] = pix_color_inside
    return ellipse


BLUE_CELLS_COLOR_DIST = {
    'hue': NormalDist(mu=184.98805646036917, sigma=13.690580569546032),
    'sat': NormalDist(mu=59.25407166123779, sigma=19.295759302753112),
    'vol': NormalDist(mu=163.071661237785, sigma=11.81532113975374)
}
BROWN_CELLS_COLOR_DIST_LEFT = {
    'hue': NormalDist(mu=18.25, sigma=8.4),
    'sat': NormalDist(mu=127.03, sigma=37.06),
    'vol': NormalDist(mu=162.7, sigma=30.06)
}
BROWN_CELLS_COLOR_DIST_RIGHT = {
    'hue': NormalDist(mu=238.5, sigma=14.5),
    'sat': NormalDist(mu=103.5, sigma=27.2),
    'vol': NormalDist(mu=108.6, sigma=26.2)
}
hue_limit_brown_l = BROWN_CELLS_COLOR_DIST_LEFT['hue'].pdf(
    BROWN_CELLS_COLOR_DIST_LEFT['hue']._mu - 2 * BROWN_CELLS_COLOR_DIST_LEFT['hue']._sigma)
hue_limit_brown_r = BROWN_CELLS_COLOR_DIST_RIGHT['hue'].pdf(
    BROWN_CELLS_COLOR_DIST_RIGHT['hue']._mu - 2 * BROWN_CELLS_COLOR_DIST_RIGHT['hue']._sigma)

hue_limit_brown_mean = (hue_limit_brown_l + hue_limit_brown_r) / 2

sat_limit_brown_l = BROWN_CELLS_COLOR_DIST_LEFT['sat'].pdf(
    BROWN_CELLS_COLOR_DIST_LEFT['sat']._mu - 2 * BROWN_CELLS_COLOR_DIST_LEFT['sat']._sigma)

sat_limit_brown_r = BROWN_CELLS_COLOR_DIST_RIGHT['sat'].pdf(
    BROWN_CELLS_COLOR_DIST_RIGHT['sat']._mu - 2 * BROWN_CELLS_COLOR_DIST_RIGHT['sat']._sigma)

sat_limit_brown_mean = (sat_limit_brown_l + sat_limit_brown_r) / 2

vol_limit_brown_l = BROWN_CELLS_COLOR_DIST_LEFT['vol'].pdf(
    BROWN_CELLS_COLOR_DIST_LEFT['vol']._mu - 2 * BROWN_CELLS_COLOR_DIST_LEFT['vol']._sigma)
vol_limit_brown_r = BROWN_CELLS_COLOR_DIST_RIGHT['vol'].pdf(
    BROWN_CELLS_COLOR_DIST_RIGHT['vol']._mu - 2 * BROWN_CELLS_COLOR_DIST_RIGHT['vol']._sigma)

vol_limit_brown_mean = (vol_limit_brown_l + vol_limit_brown_r) / 2


def check_ellipse_colors(ellipse, color):
    if ellipse['colors']:
        h, s, v = zip(*ellipse['colors'])
        median_h = median(h)
        median_s = median(s)
        median_v = median(v)
        if color == 'blue':
            # pdf_h = BLUE_CELLS_COLOR_DIST['hue'].pdf(median_h)
            pdf_h = mean([BLUE_CELLS_COLOR_DIST['hue'].pdf(pixel) for pixel in h])
            hue_limit = BLUE_CELLS_COLOR_DIST['hue'].pdf(
                BLUE_CELLS_COLOR_DIST['hue']._mu - 2 * BLUE_CELLS_COLOR_DIST['hue']._sigma)
            if pdf_h >= hue_limit:
                # pdf_s = BLUE_CELLS_COLOR_DIST['sat'].pdf(median_s)
                pdf_s = mean([BLUE_CELLS_COLOR_DIST['sat'].pdf(pixel) for pixel in s])
                sat_limit = BLUE_CELLS_COLOR_DIST['sat'].pdf(
                    BLUE_CELLS_COLOR_DIST['sat']._mu - 2 * BLUE_CELLS_COLOR_DIST['sat']._sigma)
                if pdf_s >= sat_limit:
                    # pdf_v = BLUE_CELLS_COLOR_DIST['vol'].pdf(median_v)
                    pdf_v = mean([BLUE_CELLS_COLOR_DIST['vol'].pdf(pixel) for pixel in v])
                    val_limit = BLUE_CELLS_COLOR_DIST['vol'].pdf(
                        BLUE_CELLS_COLOR_DIST['vol']._mu - 2 * BLUE_CELLS_COLOR_DIST['vol']._sigma)
                    if pdf_v >= val_limit:
                        return True, pdf_h, pdf_s, pdf_v, median_h
        if color == 'brown':
            dist = []
            pdf_h = []
            pdf_h_l = [BROWN_CELLS_COLOR_DIST_LEFT['hue'].pdf(pixel) for pixel in h]
            pdf_h_r = [BROWN_CELLS_COLOR_DIST_RIGHT['hue'].pdf(pixel) for pixel in h]
            for idx, pixel in enumerate(h):
                if pdf_h_l[idx] > pdf_h_r[idx]:
                    dist.append("left")
                    pdf_h.append(pdf_h_l[idx])
                else:
                    dist.append("right")
                    pdf_h.append(pdf_h_r[idx])
            pdf_h = mean(pdf_h)
            if pdf_h >= hue_limit_brown_mean:
                pdf_s = []
                for idx, pixel in enumerate(s):
                    if dist[idx] == 'left':
                        pdf = BROWN_CELLS_COLOR_DIST_LEFT['sat'].pdf(pixel)
                    elif dist[idx] == 'right':
                        pdf = BROWN_CELLS_COLOR_DIST_RIGHT['sat'].pdf(pixel)
                    pdf_s.append(pdf)
                pdf_s = mean(pdf_s)
                if pdf_s >= sat_limit_brown_mean:
                    pdf_v = []
                    for idx, pixel in enumerate(v):
                        if dist[idx] == 'left':
                            pdf = BROWN_CELLS_COLOR_DIST_LEFT['vol'].pdf(pixel)
                        elif dist[idx] == 'right':
                            pdf = BROWN_CELLS_COLOR_DIST_RIGHT['vol'].pdf(pixel)
                        pdf_v.append(pdf)
                    pdf_v = mean(pdf_v)
                    if pdf_v >= vol_limit_brown_mean:
                        return True, pdf_h, pdf_s, pdf_v, median_h

    return False, 0, 0, 0, 0

Buffor = 10
sat_limit_down_blue = BLUE_CELLS_COLOR_DIST['sat']._mu - 2 * BLUE_CELLS_COLOR_DIST['sat']._sigma - Buffor
sat_limit_up_blue = BLUE_CELLS_COLOR_DIST['sat']._mu + 2 * BLUE_CELLS_COLOR_DIST['sat']._sigma + Buffor

vol_limit_down_blue = BLUE_CELLS_COLOR_DIST['vol']._mu - 2 * BLUE_CELLS_COLOR_DIST['vol']._sigma - Buffor
vol_limit_up_blue = BLUE_CELLS_COLOR_DIST['vol']._mu + 2 * BLUE_CELLS_COLOR_DIST['vol']._sigma + Buffor

sat_limit_down_brown_l = BROWN_CELLS_COLOR_DIST_LEFT['sat']._mu - 2 * BROWN_CELLS_COLOR_DIST_LEFT['sat']._sigma - Buffor
sat_limit_up_brown_l = BROWN_CELLS_COLOR_DIST_LEFT['sat']._mu + 2 * BROWN_CELLS_COLOR_DIST_LEFT['sat']._sigma + Buffor

vol_limit_down_brown_l = BROWN_CELLS_COLOR_DIST_LEFT['vol']._mu - 2 * BROWN_CELLS_COLOR_DIST_LEFT['vol']._sigma - Buffor
vol_limit_up_brown_l = BROWN_CELLS_COLOR_DIST_LEFT['vol']._mu + 2 * BROWN_CELLS_COLOR_DIST_LEFT['vol']._sigma + Buffor

sat_limit_down_brown_r = BROWN_CELLS_COLOR_DIST_RIGHT['sat']._mu - 2 * BROWN_CELLS_COLOR_DIST_RIGHT['sat']._sigma - Buffor
sat_limit_up_brown_r = BROWN_CELLS_COLOR_DIST_RIGHT['sat']._mu + 2 * BROWN_CELLS_COLOR_DIST_RIGHT['sat']._sigma + Buffor

vol_limit_down_brown_r = BROWN_CELLS_COLOR_DIST_RIGHT['vol']._mu - 2 * BROWN_CELLS_COLOR_DIST_RIGHT['vol']._sigma - Buffor
vol_limit_up_brown_r = BROWN_CELLS_COLOR_DIST_RIGHT['vol']._mu + 2 * BROWN_CELLS_COLOR_DIST_RIGHT['vol']._sigma + Buffor


hue_limit_down_brown_l = BROWN_CELLS_COLOR_DIST_LEFT['hue']._mu - 2 * BROWN_CELLS_COLOR_DIST_LEFT['hue']._sigma - Buffor
hue_limit_up_brown_l = BROWN_CELLS_COLOR_DIST_LEFT['hue']._mu + 2 * BROWN_CELLS_COLOR_DIST_LEFT['hue']._sigma + Buffor

hue_limit_down_brown_r = BROWN_CELLS_COLOR_DIST_RIGHT['hue']._mu - 2 * BROWN_CELLS_COLOR_DIST_RIGHT[
    'hue']._sigma - Buffor
hue_limit_up_brown_r = BROWN_CELLS_COLOR_DIST_RIGHT['hue']._mu + 2 * BROWN_CELLS_COLOR_DIST_RIGHT['hue']._sigma + Buffor


def count_ratio(elipse):
    color_array = elipse['colors']
    vol = elipse['radius_x'] * elipse['radius_y'] * pi
    color_len = len(color_array)
    if elipse['color'] == "blue":
        filtered = [pixel for pixel in color_array if pixel[0] > 150 and pixel[0] < 290]
        filtered = [pixel for pixel in filtered if pixel[1] > sat_limit_down_blue and pixel[1] < sat_limit_up_blue]
        filtered = [pixel for pixel in filtered if pixel[2] > vol_limit_down_blue and pixel[2] < vol_limit_up_blue]
    elif elipse['color'] == "brown":
        filtered = [pixel for pixel in color_array if (
                (hue_limit_down_brown_l < pixel[0] < hue_limit_up_brown_l) or
                (hue_limit_down_brown_r < pixel[0] < hue_limit_up_brown_r))]

        filtered = [pixel for pixel in filtered if (
                (sat_limit_down_brown_r < pixel[1] < sat_limit_up_brown_r) or
                (sat_limit_down_brown_l < pixel[1] < sat_limit_up_brown_l))]

        filtered = [pixel for pixel in filtered if (
                (vol_limit_down_brown_r < pixel[2] < vol_limit_up_brown_r) or
                (vol_limit_down_brown_l < pixel[2] < vol_limit_up_brown_l))]
    div = (2*(color_len-len(filtered))) + np.random.normal(1,1)/100000000
    return len(filtered)/div # *vol  / color_len


def get_best_ellipse(ellipse, type='ratio'):
    if type == 'ratio':
        x = sorted(ellipse, key=lambda i: i['rat'] + (i['pdf_h'] * i['pdf_v']))
        x[-1]['metric_value'] = x[-1]['rat']
    if type == 'pdf_multi':
        x = sorted(ellipse, key=lambda i: i['pdf_h']  # i['pdf_s']
                                          * i['pdf_v'])
        x[-1]['metric_value'] = x[-1]['pdf_h'] * x[-1]['pdf_v']

    if type == 'median_dist':
        x = sorted(ellipse, key=lambda i: abs(i['median_v'] - BLUE_CELLS_COLOR_DIST['hue']._mu))
        x[-1]['metric_value'] = abs(x[-1]['median_v'] - BLUE_CELLS_COLOR_DIST['hue']._mu)
    return x[-1]


def select_elipses(elipses, type):
    def change_key(key, item):
        item['parent'] = key
        if key == 'pdf_h':
            item['parent_v'] = item['pdf_h'] * item['pdf_v'] * item['pdf_s']
        return item

    if type == 'pdf_h':
        elipses_fitered = [elipse.copy() for elipse in elipses if
                           elipse['pdf_h'] * elipse['pdf_v'] * elipse['pdf_s'] > 0]
    if type == 'ratio':
        # elipses_fitered = [elipse.copy() for elipse in elipses if
        #                   elipse['color_ratio'] != 0 and log(elipse['color_ratio']) >= 6]
        elipses_fitered = [elipse.copy() for elipse in elipses if
                           elipse['rat'] >= 0.3]
    if type == 'dist':
        elipses_fitered = [elipse.copy() for elipse in elipses if
                           abs(elipse['median_v'] - BLUE_CELLS_COLOR_DIST['hue']._mu) >= BLUE_CELLS_COLOR_DIST[
                               'hue']._mu - BLUE_CELLS_COLOR_DIST['hue']._sigma]

    return elipses_fitered


def ellipse_run(params):
    MEAN_ELLIPSE_PARAMS = params["MEAN_ELLIPSE_PARAMS"]
    radius_sum = int(MEAN_ELLIPSE_PARAMS['radius_x'] + MEAN_ELLIPSE_PARAMS['radius_y'])
    step = params['x_steps_size']
    y_step = params['y_step']
    ellipses_mesh = []
    rad_x = int(MEAN_ELLIPSE_PARAMS['radius_x'])
    rad_y = int(MEAN_ELLIPSE_PARAMS['radius_y'])
    for x in range(0, params['img'].width, step):
        ellipses = []
        cy = y_step
        cx = x
        tmp_ellipse = get_ellipse(cx, cy, 0, radius_sum, params['img'], radius_x=rad_x, radius_y=rad_y)
        check, pdf_h, pdf_s, pdf_v, med_v = check_ellipse_colors(tmp_ellipse, params['color'])
        # if check:
        tmp_ellipse['img_name'] = params['img_name']
        tmp_ellipse['pdf_h'] = pdf_h
        tmp_ellipse['pdf_s'] = pdf_s
        tmp_ellipse['pdf_v'] = pdf_v
        tmp_ellipse['median_v'] = med_v
        tmp_ellipse['color'] = params['color']
        tmp_ellipse['num_pixels'] = len(tmp_ellipse['colors'])

        tmp_ellipse['rat'] = count_ratio(tmp_ellipse)
        ellipses.append(tmp_ellipse)

        if x + step < params['img'].width:
            if y_step + params['y_step_size'] < params['img'].height:
                tmp_ellipse = get_ellipse(int((x + step / 2)),
                                          int(cy + params['y_step_size'] / 2),
                                          0,
                                          radius_sum,
                                          params['img'],
                                          radius_x=rad_x,
                                          radius_y=rad_y)
                check, pdf_h, pdf_s, pdf_v, med_v = check_ellipse_colors(tmp_ellipse, params['color'])
                # if check:
                tmp_ellipse['img_name'] = params['img_name']
                tmp_ellipse['pdf_h'] = pdf_h
                tmp_ellipse['pdf_s'] = pdf_s
                tmp_ellipse['pdf_v'] = pdf_v
                tmp_ellipse['median_v'] = med_v
                tmp_ellipse['color'] = params['color']

                tmp_ellipse['num_pixels'] = len(tmp_ellipse['colors'])
                tmp_ellipse['rat'] = count_ratio(tmp_ellipse)
                ellipses.append(tmp_ellipse)
        if len(ellipses) > 0:
            # tmp_ellipse = get_best_ellipse(ellipses)
            ellipses_mesh.extend(ellipses)
    ### filtering step
    ellipses_mesh_final = []
    # ellipses_mesh_final.extend(select_elipses(ellipses_mesh,'pdf_h'))
    ellipses_mesh_final.extend(select_elipses(ellipses_mesh, 'ratio'))
    # ellipses_mesh_final.extend(select_elipses(ellipses_mesh, 'dist'))
    # [elipse for elipse in ellipses_mesh if elipse['pdf_h'] !=0]
    shake_x_range = range(-int(MEAN_ELLIPSE_PARAMS['radius_x'] ), int(MEAN_ELLIPSE_PARAMS['radius_x'] ), 6)
    shake_y_range = range(-int(MEAN_ELLIPSE_PARAMS['radius_x'] ), int(MEAN_ELLIPSE_PARAMS['radius_y'] ), 6)
    deforme_range_x = range(90, 150, 40)
    deforme_range_y = range(90, 150, 40)
    rotate_range = range(0, 179, 60)
    last_elipse = None
    final_ellipses = []
    for elipse in ellipses_mesh_final:
        ellipses = []
        for x_shake in shake_x_range:
            for y_shake in shake_y_range:
                for x_deform in deforme_range_x:
                    for y_deform in deforme_range_y:
                        #for deg in rotate_range:
                            deg = 0
                            x = elipse['center_x'] + x_shake
                            y = elipse['center_y'] + y_shake
                            radius_x = elipse['radius_x'] * x_deform / 100
                            radius_y = elipse['radius_y'] * y_deform / 100
                            radius_sum = int(radius_x + radius_y)

                            tmp_ellipse = get_ellipse(x, y, deg, radius_sum, params['img'], radius_x=radius_x,
                                                      radius_y=radius_y)
                            check, pdf_h, pdf_s, pdf_v, med_v = check_ellipse_colors(tmp_ellipse, params['color'])
                            # if check:
                            if check:
                                tmp_ellipse['img_name'] = params['img_name']
                                tmp_ellipse['pdf_h'] = pdf_h
                                tmp_ellipse['pdf_s'] = pdf_s
                                tmp_ellipse['pdf_v'] = pdf_v
                                tmp_ellipse['median_v'] = med_v
                                tmp_ellipse['color'] = params['color']
                                tmp_ellipse['num_pixels'] = len(tmp_ellipse['colors'])

                                tmp_ellipse['rat'] = count_ratio(tmp_ellipse)
                                # tmp_ellipse['parent']=elipse['parent']
                                # tmp_ellipse['parent_v']=elipse['parent_v']
                                ellipses.append(tmp_ellipse)
        if len(ellipses) > 0:
            tmp_ellipse = get_best_ellipse(ellipses, type='ratio')
            tmp_ellipse['metric'] = 'ratio'
            final_ellipses.append(tmp_ellipse)
            # tmp_ellipse = get_best_ellipse(ellipses, type='pdf_multi')
            # tmp_ellipse['metric'] = 'pdf_multi'
            # final_ellipses.append(tmp_ellipse)
            # tmp_ellipse = get_best_ellipse(ellipses, type='median_dist')
            # tmp_ellipse['metric'] = 'median_dist'
            # final_ellipses.append(tmp_ellipse)
            # final_ellipses.extend(ellipses)

    return final_ellipses, ellipses_mesh_final, ellipses_mesh


def fit_new_ellipse(elipse1, elipse2, params):
    MEAN_ELLIPSE_PARAMS = params["MEAN_ELLIPSE_PARAMS"]
    img = params["img"]

    shake_x_range = range(-int(MEAN_ELLIPSE_PARAMS['radius_x'] / 3), int(MEAN_ELLIPSE_PARAMS['radius_x'] / 3), 60)
    shake_y_range = range(-int(MEAN_ELLIPSE_PARAMS['radius_x'] / 3), int(MEAN_ELLIPSE_PARAMS['radius_y'] / 3), 60)
    deforme_range_x = range(50, 150, 150)
    deforme_range_y = range(50, 150, 150)
    rotate_range = range(0, 180, 180)

    ratio_sum = elipse1['rat'] + elipse2['rat']
    rat_1 = elipse1['rat'] / ratio_sum
    rat_2 = elipse2['rat'] / ratio_sum

    new_x = int((rat_1 * elipse1['center_x'] + rat_2 * elipse2['center_x']))
    new_y = int((rat_1 * elipse1['center_y'] + rat_2 * elipse2['center_y']))

    mean_theta = (rat_1 * elipse1['theta'] + rat_2 * elipse1['theta'])
    mean_radius_x = (elipse1['radius_x'] + elipse2['radius_x']) / 2
    mean_radius_y = (elipse1['radius_y'] + elipse2['radius_y']) / 2
    radius_sum = int(mean_radius_x + mean_radius_y)
    tmp_ellipse = get_ellipse(new_x, new_y, mean_theta, radius_sum, img, radius_x=mean_radius_x, radius_y=mean_radius_y)
    ellipses = []
    best_elipse = None
    for x_shake in shake_x_range:
        for y_shake in shake_y_range:
            for x_deform in deforme_range_x:
                for y_deform in deforme_range_y:
                    for deg in rotate_range:
                        x = tmp_ellipse['center_x'] + x_shake
                        y = tmp_ellipse['center_y'] + y_shake
                        radius_x = tmp_ellipse['radius_x'] * x_deform / 100
                        radius_y = tmp_ellipse['radius_y'] * y_deform / 100
                        radius_sum = int(radius_x + radius_y)
                        new_tmp_ellipse = get_ellipse(x, y, deg, radius_sum, img, radius_x=radius_x,
                                                      radius_y=radius_y)
                        check, pdf_h, pdf_s, pdf_v, med_v = check_ellipse_colors(new_tmp_ellipse,
                                                                                 params['color'])
                        if check:
                            new_tmp_ellipse['img_name'] = img_name
                            new_tmp_ellipse['pdf_h'] = pdf_h
                            new_tmp_ellipse['pdf_s'] = pdf_s
                            new_tmp_ellipse['pdf_v'] = pdf_v
                            new_tmp_ellipse['color'] = params['color']
                            new_tmp_ellipse['median_v'] = med_v
                            new_tmp_ellipse['rat'] = count_ratio(new_tmp_ellipse)
                            ellipses.append(new_tmp_ellipse)
    if len(ellipses) > 0:
        new_tmp_ellipse = get_best_ellipse(ellipses, type='ratio')
        new_tmp_ellipse['metric'] = 'ratio'
        polygon = create_polygon(new_tmp_ellipse)
        best_elipse = (new_tmp_ellipse, polygon)
    else:
        elipse_list = [elipse1, elipse2]
        new_best = get_best_ellipse(elipse_list, type='ratio')
        polygon = create_polygon(new_best)
        best_elipse = (new_best, polygon)

    return best_elipse


def connect_clusters(clusters):
    change = True
    while (change):
        change = False
        tmp_clusters = clusters
        for idx, cluster in enumerate(tmp_clusters):
            for idx2, cluster2 in enumerate(tmp_clusters):
                connect = False
                if cluster == cluster2:
                    continue
                for cluster_item in cluster:
                    if cluster_item in cluster2:
                        connect = True
                        break
                if connect:
                    change = True
                    new_cluster = cluster
                    for item in cluster2:
                        if item not in cluster:
                            new_cluster.append(item)
                    clusters[idx] = new_cluster
                    del clusters[idx2]
                    break
            if change:
                break
        if change:
            break

    return clusters


def fit_new_ellipse_cluster(cluster, params):
    cluster = [item[0] for item in cluster]
    MEAN_ELLIPSE_PARAMS = params["MEAN_ELLIPSE_PARAMS"]
    img = params["img"]
    shake_x_range = range(-int(MEAN_ELLIPSE_PARAMS['radius_x'] / 3), int(MEAN_ELLIPSE_PARAMS['radius_x'] / 3), 3)
    shake_y_range = range(-int(MEAN_ELLIPSE_PARAMS['radius_x'] / 3), int(MEAN_ELLIPSE_PARAMS['radius_y'] / 3), 3)
    deforme_range_x = range(70, 130, 20)
    deforme_range_y = range(70, 130, 20)
    # rotate_range = range(0, 179, 180)
    rad_x = int(MEAN_ELLIPSE_PARAMS['radius_x'])
    rad_y = int(MEAN_ELLIPSE_PARAMS['radius_y'])
    radius_sum = rad_x + rad_y
    ratio_sum = sum([elipse['rat'] for elipse in cluster])
    rats = [elipse['rat'] / ratio_sum for elipse in cluster]
    pos_x = 0
    pos_y = 0
    for idx, elipse in enumerate(cluster):
        pos_x += rats[idx] * elipse['center_x']
        pos_y += rats[idx] * elipse['center_y']
    pos_x = int(pos_x)
    pos_y = int(pos_y)

    tmp_ellipse = get_ellipse(pos_x, pos_y, 0, radius_sum, img, radius_x=rad_x, radius_y=rad_y)
    ellipses = []
    for x_shake in tqdm(shake_x_range, desc='x shake', leave=True, position=0):
        for y_shake in tqdm(shake_y_range, desc='y shake', leave=True, position=0):
            for x_deform in tqdm(deforme_range_x, desc='x_deform', leave=True, position=0):
                for y_deform in tqdm(deforme_range_y, desc='y_deform', leave=True, position=0):
                    # for deg in tqdm(rotate_range,desc="rotate", leave=True,position=0):
                    deg = 0
                    x = tmp_ellipse['center_x'] + x_shake
                    y = tmp_ellipse['center_y'] + y_shake
                    radius_x = tmp_ellipse['radius_x'] * x_deform / 100
                    radius_y = tmp_ellipse['radius_y'] * y_deform / 100
                    radius_sum = int(radius_x + radius_y)
                    new_tmp_ellipse = get_ellipse(x, y, deg, radius_sum, img, radius_x=radius_x,
                                                  radius_y=radius_y)
                    check, pdf_h, pdf_s, pdf_v, med_v = check_ellipse_colors(new_tmp_ellipse,
                                                                             params['color'])
                    if check:
                        new_tmp_ellipse['img_name'] = img_name
                        new_tmp_ellipse['pdf_h'] = pdf_h
                        new_tmp_ellipse['pdf_s'] = pdf_s
                        new_tmp_ellipse['pdf_v'] = pdf_v
                        new_tmp_ellipse['color'] = params['color']
                        new_tmp_ellipse['median_v'] = med_v
                        new_tmp_ellipse['num_pixels'] = len(new_tmp_ellipse['colors'])
                        new_tmp_ellipse['rat'] = count_ratio(new_tmp_ellipse)
                        ellipses.append(new_tmp_ellipse)
    best_elipse = None
    if len(ellipses) > 0:
        new_tmp_ellipse = get_best_ellipse(ellipses, type='ratio')
        new_tmp_ellipse['metric'] = 'ratio'
        polygon = create_polygon(new_tmp_ellipse)
        best_elipse = (new_tmp_ellipse, polygon)

    return best_elipse


def create_elipses_from_intersects(intersects_list, params):
    clusters = []
    for intersection in intersects_list:
        ellipse1 = intersection[0]
        ellipse2 = intersection[1]
        found = False
        for idx, cluster in enumerate(clusters):
            for cluster_item in cluster:
                if cluster_item == ellipse1:
                    if ellipse2 not in cluster:
                        cluster.append(ellipse2)
                    found = True
                    break
                if cluster_item == ellipse2:
                    if ellipse1 not in cluster:
                        cluster.append(ellipse1)
                    found = True
                    break
        if not found:
            clusters.append([ellipse1, ellipse2])

            # if len(cluster)>=3:
            #    new_ellipse = fit_new_ellipse_cluster(cluster,params)
            #   clusters[idx]=[new_ellipse]

    clusters = connect_clusters(clusters)
    return clusters


def filter_elipses(intersected, params):
    filtered = []
    for pair in intersected:
        e1 = pair[0]
        e2 = pair[1]
        new_e = fit_new_ellipse_cluster([e1, e2], params)
        elipses = [e1, e2]
        if new_e:
            elipses.append(new_e)
            x = sorted(elipses, key=lambda i: i[0]['rat'] / len(i[0]["colors"]))
            best = x[-1]
            if new_e != best:
                filtered.append(e1)
                filtered.append(e2)
            else:
                filtered.append(best)
        else:
            x = sorted(elipses, key=lambda i: i[0]['rat'] / len(i[0]["colors"]))
            best = x[-1]
            filtered.append(best)
    return filtered


def overlaping_transform(ellipses, params):
    final_list = []
    if len(ellipses) > 0:
        polygons = [create_polygon(ellipse) for ellipse in ellipses]
        elipses_list_check = list(zip(ellipses, polygons))
        intersects_list = []
        not_overlaping = []
        for elipse in elipses_list_check:
            intersected = False
            for elipse2 in elipses_list_check:
                if elipse == elipse2:
                    continue
                if elipse[1].intersects(elipse2[1]):
                    tup = (elipse, elipse2)
                    tup2 = (elipse2, elipse)
                    if tup not in intersects_list and tup2 not in intersects_list:
                        intersects_list.append((elipse, elipse2))
                    intersected = True
            if not intersected:
                not_overlaping.append(elipse)
        ###
        strong_limit = 0.70
        weak_elipses = [pair for pair in intersects_list if
                        (pair[0][0]['rat'] < strong_limit and pair[1][0]['rat'] < strong_limit)]
        stronk_elipses = [pair for pair in intersects_list if
                          (pair[0][0]['rat'] >= strong_limit or pair[1][0]['rat'] >= strong_limit)]
        for pair in stronk_elipses:
            if pair[0][0]['rat'] >= strong_limit:
                not_overlaping.append(pair[0])
            if pair[1][0]['rat'] >= strong_limit:
                not_overlaping.append(pair[1])
        clusters = create_elipses_from_intersects(weak_elipses, params)
        for cluster in clusters:
            potencial_better = fit_new_ellipse_cluster(cluster, params)
            if potencial_better:
                not_overlaping.append(potencial_better)
        # not_overlaping.extend(elipses)
        final_list = [elipse[0] for elipse in not_overlaping]
    return final_list


def check_overlapping(ellipses):
    for combo in list(combinations(ellipses, 2)):
        if combo[0][1].intersects(combo[1][1]):
            return True
    return False


def create_polygon(ellipse):
    ellipse1 = Ellipse((ellipse["center_x"], ellipse["center_y"]), ellipse["radius_x"] * 2, ellipse["radius_y"] * 2,
                       ellipse["theta"])
    vertices = ellipse1.get_verts()
    polygon = Polygon(vertices)
    return polygon


def experiment(img_name, color,prefix=None):
    if prefix:
        prefix_title = prefix
    else:
        prefix_title = ""
    img_path = os.path.join(str(data_path), str(img_name) + ".jpg")
    with Image.open(img_path) as img:
        img = img.convert('HSV')
        results = []
        mesh_final = []
        mesh = []
        if color == "blue":
            x_steps_size = 2 * int(MEAN_ELLIPSE_PARAMS_BLUE['radius_x'])
            y_step_size = 2 * int(MEAN_ELLIPSE_PARAMS_BLUE['radius_y'])
            MEAN_ELLIPSE_PARAMS = MEAN_ELLIPSE_PARAMS_BLUE
        elif color == "brown":
            x_steps_size = 2 * int(MEAN_ELLIPSE_PARAMS_BROWN['radius_x'])
            y_step_size = 2 * int(MEAN_ELLIPSE_PARAMS_BROWN['radius_y'])
            MEAN_ELLIPSE_PARAMS = MEAN_ELLIPSE_PARAMS_BROWN
        params_combination = []
        assert MEAN_ELLIPSE_PARAMS
        for y_step in range(0, img.height, y_step_size):
            params = {
                'x_steps_size': x_steps_size,
                'y_step': y_step,
                'y_step_size': y_step_size,
                'img_name': img_name,
                'img': img,
                'color': color,
                'MEAN_ELLIPSE_PARAMS': MEAN_ELLIPSE_PARAMS
            }
            params_combination.append(params)
        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
            results_exe = executor.map(ellipse_run, params_combination)
            for res in tqdm(results_exe, total=len(params_combination), desc=f"{img_name} progress"):
                results.extend(res[0])
                mesh_final.extend(res[1])
                mesh.extend(res[2])
    params = {
        'color': color,
        'img': img,
        'img_name': img_name,
        'MEAN_ELLIPSE_PARAMS': MEAN_ELLIPSE_PARAMS
    }
    res = pd.DataFrame(results)
    mesh = pd.DataFrame(mesh)
    mesh_final = pd.DataFrame(mesh_final)
    with open(f"results_elipses/results_{img_name}_{color}_{prefix_title}.pickle", 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"results_elipses/results_{img_name}_mesh_{color}_{prefix_title}.pickle", 'wb') as handle:
        pickle.dump(mesh, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"results_elipses/results_{img_name}_meshFinal_{color}_{prefix_title}.pickle", 'wb') as handle:
        pickle.dump(mesh_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # not_overlaping = overlaping_transform(results,params)
    # overlap = pd.DataFrame(not_overlaping)
    # with open(f"results_elipses/results_{img_name}_overlap.pickle", 'wb') as handle:
    #    pickle.dump(overlap, handle, protocol=pickle.HIGHEST_PROTOCOL)


def overlap_exp(img_name, color,prefix=None):
    if prefix:
        prefix_title = prefix
    else:
        prefix_title = ""
    img_path = os.path.join(str(data_path), str(img_name) + ".jpg")
    with Image.open(img_path) as img:
        img = img.convert('HSV')
        if color == "blue":
            MEAN_ELLIPSE_PARAMS = MEAN_ELLIPSE_PARAMS_BLUE
        elif color == "brown":
            MEAN_ELLIPSE_PARAMS = MEAN_ELLIPSE_PARAMS_BROWN
        assert MEAN_ELLIPSE_PARAMS

        params = {
            'color': color,
            'img': img,
            'img_name': img_name,
            'MEAN_ELLIPSE_PARAMS': MEAN_ELLIPSE_PARAMS
        }
    results_file = f"../src/results_elipses/results_{img_name}_meshFinal_{color}_{prefix_title}.pickle"
    assert os.path.exists(results_file)
    with open(results_file, 'rb') as handle:
        results = pickle.load(handle)
    results = list(results.T.to_dict().values())
    not_overlaping = overlaping_transform(results, params)
    overlap = pd.DataFrame(not_overlaping)
    with open(f"results_elipses/results_{img_name}_overlap_{color}_{prefix_title}.pickle", 'wb') as handle:
        pickle.dump(overlap, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    img_list = ["p1_0299_6","p1_0308_2","p1_0309_2","p1_0309_5","p1_0309_11","p1_0317_3","p1_0299_1_z_testowego"]
    all = False
    if all:
        img_list = []
        for root, dirs, files in os.walk(train_set, topdown=False):
            for name in files:
                if name.endswith('.json'):
                    img_list.append(name.split('.')[0])
    print("Image count : ", len(img_list))
    for img_name in tqdm(img_list, colour='blue', desc='Images'):
        prefix = 'big shake'
        experiment(img_name,'brown',prefix)
        overlap_exp(img_name, 'brown',prefix)
        experiment(img_name, 'blue',prefix)
        overlap_exp(img_name, 'blue',prefix)
