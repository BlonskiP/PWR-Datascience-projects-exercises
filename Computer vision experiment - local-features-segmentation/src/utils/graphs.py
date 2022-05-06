import networkx as nx
import random
import cv2 as cv
import numpy as np
import sys
from itertools import combinations
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
def random_graph(num_nodes=5,prob_edge=0.3,seed=42):
    '''

    :param num_nodes: int - number of nodes
    :param prob_edge: float - probablity of edge acc
    :param seed: int - seed for random number generator
    :return: networkx Graph - for test purpose
    '''
    random.seed(seed)
    G = nx.fast_gnp_random_graph(num_nodes, prob_edge, seed=seed, directed=False)
    center = len(G._node)
    for idx, node in enumerate(G._node):
        G._node[idx] = {'x': center+random.randint(-center,center),
                        'y': center+random.randint(-center,center)}
    return G

def get_detector(name='sift',nFeatures=None):
    if name=='sift':
        detector = cv.SIFT_create(nFeatures)
        return detector
    elif name=='orb':
        detector = cv.ORB_create(nFeatures)
        return detector
    else:
        return name

def get_keypoints(img,nFeatures=None,return_pixel_color=None,detector='sift'):
    detector = get_detector(detector,nFeatures)
    kp, des = detector.detectAndCompute(img, None)

    if return_pixel_color:
        key_points = []
        pixels = []
        for keypoint in kp:
            x,y = keypoint.pt
            x= int(x)
            y = int(y)
            if keypoint.pt not in key_points:
                key_points.append(keypoint.pt)
                pixels.append(img[x][y].tolist())

        return zip(key_points,pixels)
    else:
        key_points = [key.pt for key in kp]
    #keyrelative_neighborhood_points = np.unique(key_points,axis=0)

    return key_points

def get_distances_between_keypoints(keypoints):
    return cdist(keypoints, keypoints)

def relative_neighborhood_baseline(keypoints):
    G = nx.empty_graph()
    for idx, key in enumerate(keypoints):
        G.add_node(idx)
        G._node[idx]={'x':key[0],
                      'y':key[1]}
    distance_matrix = get_distances_between_keypoints(keypoints)
    comb = list(combinations(list(G.nodes),2))
    #print(comb)
    for combination in tqdm(comb):
        can_add = True
        #print(combination[0])
        node_q = combination[0]
        node_p = combination[1]

        distances_q = distance_matrix[node_q]
        distances_p = distance_matrix[node_p]

        dist_q_p = distances_q[node_p]
        # from q
        closer_nodes_q = np.where(distances_q < dist_q_p)[0]
        # from p
        closer_nodes_p = np.where(distances_p < dist_q_p)[0]
        #print('distance', dist_q_p)
        for closer_node in closer_nodes_q:
            if closer_node in closer_nodes_q and closer_node in closer_nodes_p:
                #print(closer_nodes_p)
                #print(closer_nodes_q)
                can_add = False
                break
        if can_add:
            #print('edge between', node_q,node_p,'distance',dist_q_p)
            G.add_edge(node_q,node_p)
    #print(distance_matrix)
    return G


def delaunay_triangulation(keypoints):
    tri = Delaunay(keypoints)
    G = nx.empty_graph()
    for idx, key in enumerate(keypoints):
        G.add_node(idx)
        G._node[idx] = {'x': key[0],
                        'y': key[1]}
    print("delaunay_triangulation")
    for vertexes in tqdm(tri.simplices):
        vertexes = list(vertexes)
        G.add_edge(vertexes[0], vertexes[1])
        G.add_edge(vertexes[1], vertexes[2])
        G.add_edge(vertexes[2], vertexes[0])
    return G

def delete_not_RNG_edges(G,keypoints=None):
    vertices = keypoints
    G_tmp = G.copy()
    if keypoints is None:
        vertices = []
        for idx, key in enumerate(G_tmp._node):
            x = G_tmp._node[idx]['x']
            y = G_tmp._node[idx]['y']
            vertices.append([x,y])
    distance_matrix = get_distances_between_keypoints(vertices)
    edges_to_delete = []
    print("edges cleaning")
    for edge in tqdm(G.edges):
        node_q = edge[0]
        node_p = edge[1]
        distances_q = distance_matrix[node_q]
        distances_p = distance_matrix[node_p]
        dist_q_p = distances_q[node_p]
        # from q
        closer_nodes_q = np.where(distances_q < dist_q_p)[0]
        # from p
        closer_nodes_p = np.where(distances_p < dist_q_p)[0]
        for closer_node in closer_nodes_q:
            if closer_node in closer_nodes_q and closer_node in closer_nodes_p:
                #print(closer_nodes_p)
                #print(closer_nodes_q)
                edges_to_delete.append(edge)
                break
    for edge in edges_to_delete:
        node_q = edge[0]
        node_p = edge[1]
        G_tmp.remove_edge(node_q,node_p)
    return G_tmp

def relative_neighborhood(keypoints):
    dt_graph = delaunay_triangulation(keypoints)
    g = delete_not_RNG_edges(dt_graph,keypoints)
    return g
