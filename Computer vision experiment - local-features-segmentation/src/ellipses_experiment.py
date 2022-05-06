import concurrent.futures
import os
import pickle

import matplotlib.image as mpimg
import pandas as pd
from PIL import Image
from tqdm import tqdm
from math import sin, cos
import statistics
from statistics import NormalDist, mean

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
    'blue':NormalDist(mu=162.8042662974839, sigma=14.133422501229884),
    'red':NormalDist(mu=147.89261081749171, sigma=29.41359934013295),
    'green':NormalDist(mu=133.23393004166803, sigma=30.903213662200027),
}
BROWN_CELLS_COLOR_DIST = {
    'blue':NormalDist(mu=88.10424552490524, sigma=24.622359440726957),
    'red':NormalDist(mu=157.64154611480924, sigma=25.734934966067815),
    'green':NormalDist(mu=117.84177901266865, sigma=29.850845608312493),
}
b1_min,b1_max,r1_min,r1_max,g1_min,g1_max =[84, 211, 40, 242, 33, 234]

def check_ellipse_colors(ellipse,color):
    r,g,b = zip(*ellipse['colors'])
    mean_r = mean(r)
    mean_g = mean(g)
    mean_b = mean(b)
    if color == 'blue':
        pdf_b = BLUE_CELLS_COLOR_DIST['blue'].pdf(mean_b)
        blue_limit=BLUE_CELLS_COLOR_DIST['blue'].pdf(BLUE_CELLS_COLOR_DIST['blue']._mu-BLUE_CELLS_COLOR_DIST['blue']._sigma)
        if pdf_b >= blue_limit:
            red_limit = BLUE_CELLS_COLOR_DIST['red'].pdf(BLUE_CELLS_COLOR_DIST['red']._mu-BLUE_CELLS_COLOR_DIST['red']._sigma)
            pdf_r = BLUE_CELLS_COLOR_DIST['red'].pdf(mean_r)
            if pdf_r >= red_limit:
                green_limit  = BLUE_CELLS_COLOR_DIST['green'].pdf(BLUE_CELLS_COLOR_DIST['green']._mu-BLUE_CELLS_COLOR_DIST['green']._sigma)
                pdf_g = BLUE_CELLS_COLOR_DIST['green'].pdf(mean_g)
                if pdf_g >= green_limit:
                    return True, pdf_b
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

    img_list = ["p1_0299_6","p1_0308_2","p3_0286_1"]
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