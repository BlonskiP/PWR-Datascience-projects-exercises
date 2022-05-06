import networkx as nx
import matplotlib.pyplot as plt

#EDGE LIST PLOTING
from matplotlib import cm


def plot_Network(Graph):
    # write edgelist to grid.edgelist
    nx.write_edgelist(Graph, path="grid.edgelist", delimiter=":")
    # read edgelist from grid.edgelist
    H = nx.read_edgelist(path="grid.edgelist", delimiter=":")

    nx.draw(H,with_labels=True)
    plt.show()
def visualize(followers_network,node_groups):

    size = len(node_groups)
    print('visualizing GROUPS: ',size)
    node_size=1000/size
    color_map = []
    for node in followers_network:
        color=0
        for group in node_groups:
            color+=1
            if node in group:
                color_map.append(256/size * color)
    #pos = nx.spring_layout(followers_network)
    #pos = nx.spectral_layout(followers_network) #strict line
    pos = nx.kamada_kawai_layout(followers_network)
    #pos = nx.circular_layout(followers_network)
    nx.draw(followers_network,node_size=node_size,node_color=color_map,pos=pos)
    labels = nx.draw_networkx_labels(followers_network, pos=pos,font_size=5,alpha=0.7)
    plt.savefig('plot.png')
    plt.show()


