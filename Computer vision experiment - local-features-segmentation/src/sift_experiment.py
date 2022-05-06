import concurrent.futures
import sys; sys.path.append('../')
import cv2 as cv
import pandas as pd
import os
import pickle
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tqdm import tqdm


JSON_EXT = ".json"
JPG_EXT = ".jpg"
train_set = r"../data/Ki67/SHIDC-B-Ki-67/Train"
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams['figure.dpi'] = 120
color = {
        1:"brown",
        2:"blue",
        3:"green",
        4:"red"
}

data_path = "../data/Ki67/SHIDC-B-Ki-67/Train"



def sift_metric_keypointed(df, keypoints, radius=2, plot=False):
    df_results = df.copy()
    df_results['keypointed'] = False
    df_results['keypointed_num'] = 0
    df_keys = list(zip(df.x, df.y))
    keypoints_results = pd.DataFrame(keypoints, columns={'x': [], 'y': []})
    keypoints_results['class'] = 4
    keypoints_results['labeled'] = False
    keypoints_results['num_points'] = 0
    distances = cdist(df_keys, keypoints)
    for idx, keydf in enumerate(df_keys):  # distance from each point loop dfpoints
        dist = distances[idx]
        for keypoint_idx, distance_to_dfnode in enumerate(dist):  # loop over keypoints
            dfxy = (df_results.at[idx, 'x'], df_results.at[idx, 'y'])
            keyxy = (keypoints_results.at[keypoint_idx, 'x'], keypoints_results.at[keypoint_idx, 'y'])
            # print(dfxy,' to ', keyxy,
            #      ' dist ', cdist([dfxy],[keyxy]),' dist counted ', distance_to_dfnode, ' t ', cdist([dfxy],[keyxy])==distance_to_dfnode, ' min ', dist.min() )
            if distance_to_dfnode < radius:
                # print('small distnace ',distance_to_dfnode)
                label = df_results.at[idx, 'label_id']
                df_results.at[idx, 'keypointed'] = True
                df_results.at[idx, 'keypointed_num'] += 1
                keypoints_results.at[keypoint_idx, 'class'] = label
                keypoints_results.at[keypoint_idx, 'labeled'] = True
                keypoints_results.at[keypoint_idx, 'num_points'] += 1

                if plot:
                    plt.plot([dfxy[0], keyxy[0]], [dfxy[1], keyxy[1]], marker='*', color=color[label])
                    plt.plot(keyxy[0], keyxy[1], marker='^', color=color[label], markersize=5)

    return df_results, keypoints_results


def sift_metric_research(df, keypoints, radius=2, plot=False, img_plot=None):
    if plot:
        plt.imshow(img_plot)
    res_df, res_key = sift_metric_keypointed(df, keypoints, radius, plot=plot)
    res_count = res_df.groupby(['keypointed'], as_index=False).count()
    negall = res_count.iloc[0].x / len(df)
    posall = res_count.iloc[1].x / len(df)

    num_count = res_key.groupby(['num_points'], as_index=False).count()

    res = {
        'all_df_points': len(df),
        'all_keypoints': len(keypoints),
        'negall': negall,
        'posall': posall,
        'good_keypoints': len(res_key[res_key['labeled'] == True]),
        'bad_keypoints': len(res_key[res_key['labeled'] == False]),
        'more_good_all_df': len(df) < len(res_key[res_key['labeled'] == True]),
        'radius': radius,
    }
    if plot:
        plt.show()
    return res, res_df, res_key


def add_columns(params,df):
    #[sigma, octave, edge_limit, contrast, radius]
    df['sigma'] = params['sigma']
    df['octave'] = params['octave']
    df['edge_limit'] = params['edge_limit']
    df['contrast'] = params['contrast']
    df['radius'] = params['radius']
    df['img_name'] = params['img_name']
    df['size'] = params['size']
    return df


def sift_run(params):
    try:
        sift = cv.SIFT_create(
            nOctaveLayers=params['octave'],
            contrastThreshold=params['contrast'],
            edgeThreshold=params['edge_limit'],
            sigma=params['sigma'])
        keypoints = sift.detect(params['img'], None)
        keypoints = [keypoint for keypoint in keypoints if keypoint.size > params['size']]
        print('keypoints: ',len(keypoints),params['img_name'])
        kp = list(np.unique([key.pt for key in keypoints ], axis=0))
        res, res_df, res_key = sift_metric_research(params['df'], kp, params['radius'])
        res = add_columns(params,res)
        res_df = add_columns(params,res_df)
        res_key = add_columns(params,res_key)
    except Exception as e:
        print('error with:', e)
        print(f"params: ",params['img_name'],params['sigma'])
        return None
    else:
        #print(f"Done params:{params}")
        return res, res_df, res_key


def experiment(img_name):
    img_path = os.path.join(str(data_path), str(img_name) + ".jpg")
    json_path = os.path.join(str(data_path), img_name + ".json")
    df = pd.read_json(json_path)
    img = mpimg.imread(img_path)
    results = pd.DataFrame()
    results_df = pd.DataFrame()
    results_key = pd.DataFrame()
    #radius_range = range(2, 5, 3)
    #sigma_range = range(10, 12)
    #octave_range = range(3, 6, 5)
    #edge_Threshold_range = range(10, 11, 10)
    #contrast_range = np.arange(0.05, 0.04, -0.02)
    ##
    #radius_range=range(2,10,3) #exp1
    #sigma_range=range(11,17) #exp1
    #octave_range=range(3,54,5) #exp1
    #edge_Threshold_range = range(10,50,10) #exp1
    #contrast_range= np.arange (0.05, 0.01, -0.03) #exp1

    #radius_range=range(8,9) #exp2
    #sigma_range=range(13,14) #exp2
    #octave_range=range(30,50,5) #exp2
    #edge_Threshold_range = range(10,50,10) #exp1
    #contrast_range= np.arange (0.05, 0.01, -0.01) #exp1
    params_dic = {
    'radius' : [8],#range(8,9),
    'sigma' : range(13,14),
    'octave_range' : range(30,45,5),
    'edge_Threshold_range': [30], #range(30,50,10),
    'contrast_range' : [0.01,0.02],#np.arange (0.02, 0.009, -0.01),
    'radius_range' : [5,8],#range(5,9),
    'size' : range(8,100,10)
    }
    #radius_range= #best
    #sigma_range=range(13,14) #best
    #octave_range=range(30,45,5) #best
    #edge_Threshold_range = range(30,50,10) #best
    #contrast_range= np.arange (0.02, 0.009, -0.01) #best



    params_combination = []
    for sigma in tqdm(params_dic['sigma']):
        sigma = sigma / 10
        for octave in params_dic['octave_range']:
            for edge_limit in params_dic['edge_Threshold_range']:
                for contrast in params_dic['contrast_range']:
                    for radius in params_dic['radius_range']:
                        for size in params_dic['size']:
                            params = {
                            'radius':radius,
                            'contrast':contrast,
                            'edge_limit':edge_limit,
                            'octave':octave,
                            'size':size,
                            'sigma':sigma,
                            'img_name':img_name,
                            'df':df,
                            'img':img,
                            }
                            params_combination.append(params)
    #print(len(params_combination))
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        results_exe = executor.map(sift_run, params_combination)
        for res in tqdm(results_exe,total=len(params_combination),desc=f"{img_name} progress"):
            results.append(res)
    with open(f"results/results_{img_name}.pickle", 'wb') as handle:
        pickle.dump(list(results), handle, protocol=pickle.HIGHEST_PROTOCOL)

#img_path = os.path.join(str(data_path),str(img_name)+".jpg")
#json_path = os.path.join(str(data_path),img_name+".json")
#df = pd.read_json(json_path)
#img = mpimg.imread(img_path)
#sift = cv.SIFT_create()
#keypoints, desc = sift.detectAndCompute(img,None)
#kp = list(np.unique([key.pt for key in keypoints],axis=0))

if __name__ == '__main__':

    img_list = ["p1_0299_6",
                "p1_0308_6",
                "p1_0312_6",
                "p1_0312_3",
                "p1_0308_7",
                "p3_0221_7",
                "p3_0286_1",
                "p5_0097_3"
                ]
    all = True
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