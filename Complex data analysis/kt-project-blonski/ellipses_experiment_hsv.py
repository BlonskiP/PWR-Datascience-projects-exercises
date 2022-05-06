import concurrent.futures
import os
import pickle

import matplotlib.image as mpimg
import pandas as pd
from PIL import Image
from tqdm import tqdm
from math import sin, cos
import statistics
from statistics import NormalDist, mean,median

MEAN_ELLIPSE_PARAMS = {
    'radius_x':18,
    'radius_y':20,
}
data_path = "../data/Ki67/SHIDC-B-Ki-67/Train"
train_set = r"../data/Ki67/SHIDC-B-Ki-67/Train"

def check_ellipse_rotated(elipse_params,center_x,center_y, x, y,theta):
    radius_x = elipse_params['radius_x']
    radius_y = elipse_params['radius_y']

    angle = theta

    ellipse_eq_1 = (((x - center_x) * cos(angle) + (y - center_y) * sin(angle)) ** 2) / (radius_x) ** 2
    ellipse_eq_2 = (((x - center_x) * sin(angle) - (y - center_y) * cos(angle)) ** 2) / (radius_y) ** 2
    if ellipse_eq_1 + ellipse_eq_2 <= 1:
        return True
    else:
        return False

def get_ellipse(center_x,center_y,theta,radius_sum,img):
    start_x = center_x - radius_sum
    start_y = center_y - radius_sum
    if start_x < 0:
        start_x = 0
    if start_y < 0:
        start_y = 0
    end_x = center_x + radius_sum
    end_y = center_y + radius_sum
    if end_x > img.width:
        end_x=img.width
    if end_y > img.height:
        end_y = img.height
    pix_color_inside=[]
    x_range = range(start_x, end_x)
    y_range = range(start_y, end_y)

    for x in x_range:
        for y in y_range:
            if check_ellipse_rotated(MEAN_ELLIPSE_PARAMS,center_x,center_y, x, y,theta):
                pix_color_inside.append(img.getpixel((x, y)))
    ellipse = {
        'center_x':center_x,
        'center_y':center_y,
        'theta':theta,
        'colors':pix_color_inside,
        'radius_x':MEAN_ELLIPSE_PARAMS['radius_x'],
        'radius_y':MEAN_ELLIPSE_PARAMS['radius_y'],
        'label':1
    }
    return ellipse

BLUE_CELLS_COLOR_DIST = {
    'hue':NormalDist(mu=181.82193268186754, sigma=19.730562858306044),
    'sat':NormalDist(mu=53.71444082519001, sigma=15.51944939029337),
    'vol':NormalDist(mu=165.13680781758958, sigma=12.44291033504001)
}



def check_ellipse_colors(ellipse,color):
    h,s,v = zip(*ellipse['colors'])
    median_h = median(h)
    median_s = median(s)
    median_v = median(v)
    if color == 'blue':
        pdf_h = BLUE_CELLS_COLOR_DIST['hue'].pdf(median_h)
        hue_limit=BLUE_CELLS_COLOR_DIST['hue'].pdf(BLUE_CELLS_COLOR_DIST['hue']._mu-2*BLUE_CELLS_COLOR_DIST['hue']._sigma)
        if pdf_h >= hue_limit:
            pdf_s = BLUE_CELLS_COLOR_DIST['sat'].pdf(median_s)
            sat_limit = BLUE_CELLS_COLOR_DIST['sat'].pdf(BLUE_CELLS_COLOR_DIST['sat']._mu - 2*BLUE_CELLS_COLOR_DIST['sat']._sigma)
            if pdf_s >= sat_limit:
                pdf_v = BLUE_CELLS_COLOR_DIST['vol'].pdf(median_v)
                val_limit = BLUE_CELLS_COLOR_DIST['vol'].pdf(
                    BLUE_CELLS_COLOR_DIST['vol']._mu -2*BLUE_CELLS_COLOR_DIST['vol']._sigma)
                if pdf_v >= val_limit:
                    return True, 2*pdf_h+pdf_s+pdf_v
    return False, 0

def get_best_ellipse(ellipse):
    x = sorted(ellipse, key=lambda i: i['pdf'])
    return x[-1]


def ellipse_run(params):
    radius_sum = int(MEAN_ELLIPSE_PARAMS['radius_x'] + MEAN_ELLIPSE_PARAMS['radius_y'])
    step = params['x_steps_size']
    y_step  = params['y_step']
    theta_range = range(0,360,60)
    final_ellipses = []
    # shake
    x_shake = range(0, 5, 3)
    y_shake = range(0, 5, 3)
    for x in range(0,params['img'].width,step):
        ellipses = []
        for theta in theta_range:
            theta_to_rad = theta*0.261799388
            for x_sh in x_shake:
                for y_sh in y_shake:
                    cy = y_step+y_sh
                    cx = x+x_sh
                    tmp_ellipse = get_ellipse(cx,cy,theta_to_rad,radius_sum,params['img'])
                    check, pdf = check_ellipse_colors(tmp_ellipse,'blue')
                    if check:
                        tmp_ellipse['img_name'] = params['img_name']
                        tmp_ellipse['pdf']=pdf
                        ellipses.append(tmp_ellipse)
        if len(ellipses) >0:
            tmp_ellipse = get_best_ellipse(ellipses)
            final_ellipses.append(tmp_ellipse)
    return final_ellipses




def experiment(img_name):
    img_path = os.path.join(str(data_path), str(img_name) + ".jpg")
    with Image.open(img_path) as img:
        img = img.convert('HSV')
        results = []
        x_steps_size = int(MEAN_ELLIPSE_PARAMS['radius_x'])
        y_step_size = int(MEAN_ELLIPSE_PARAMS['radius_y'])
        params_combination=[]
        for y_step in range(0,img.height,y_step_size):
            params = {
                'x_steps_size': x_steps_size,
                'y_step':y_step,
                'img_name':img_name,
                'img':img
            }
            params_combination.append(params)
        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
            results_exe = executor.map(ellipse_run, params_combination)
            for res in tqdm(results_exe, total=len(params_combination), desc=f"{img_name} progress"):
                results.extend(res)
    res = pd.DataFrame(results)
    with open(f"results_elipses/results_{img_name}.pickle", 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    img_list = ["p1_0299_6","p1_0320_2","p2_0251_7"]
    all = False
    if all:
        img_list = []
        for root, dirs, files in os.walk(train_set, topdown=False):
            for name in files:
                if name.endswith('.json'):
                    img_list.append(name.split('.')[0])
    print("Image count : ",len(img_list))
    for img_name in tqdm(img_list,colour='blue',desc='Images'):
        try:
            experiment(img_name)
        except Exception as e:
            print('error: ',e)