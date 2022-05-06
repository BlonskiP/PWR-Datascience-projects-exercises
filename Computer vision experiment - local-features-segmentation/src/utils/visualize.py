from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


def draw_graph(G, radius=1, img_orginal=None, line_power=1) -> np.array:
    '''
    :param G: networkx graph
    :param radius: int - radius how big must be the plotted node
    :param img: PIL Image - on which we will plot graph
    :return: numpy array with image
    '''
    size = 2 * len(G.nodes)
    if img_orginal:
        img = img_orginal.copy()
    else:
        img = Image.new('RGB', (size, size))
    draw = ImageDraw.Draw(img)
    points = []
    edges = []
    # nodes
    for idx, node in enumerate(G.nodes):
        # center
        x = G.nodes[idx]['x']
        y = G.nodes[idx]['y']
        points_left = (x - radius, y - radius)
        points_right = (x + radius, y + radius)
        points.append([points_left, points_right])
    # edges
    for e in G.edges:
        node_A = G.nodes[e[0]]

        node_B = G.nodes[e[1]]
        edges.append([node_A['x'], node_A['y'], node_B['x'], node_B['y']])

    for edge in edges:
        draw.line(edge, fill=(255, 0, 0, 255), width=line_power)

    for points_pair in points:
        draw.ellipse(points_pair, fill=(255, 255, 255, 255))

    arr = np.asarray(img)
    img.close()
    imshow(arr)
    return arr
    pass


def plot_voronoi(voronoi, img_arr, points_size=1, line_width=0.5, show_points=False, show_verticles=False,save_as_file=False,filename=''):
    fig, ax = plt.subplots()
    ax.imshow(img_arr)
    fig = voronoi_plot_2d(voronoi, ax=ax, line_width=line_width, show_points=show_points, show_vertices=show_verticles,
                          point_size=points_size)

    if save_as_file:
        #plt.show()
        plt.savefig(filename)
    axes = plt.gca()
    shape = img_arr.shape
    axes.set_xlim([0, shape[0]])
    axes.set_ylim([0, shape[1]])
    axes.invert_yaxis()
    return fig
