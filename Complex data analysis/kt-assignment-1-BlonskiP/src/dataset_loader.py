from src.graph import create_graph, load_graph
import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split

seed = 42

# https://networkx.org/documentation/stable/reference/algorithms/centrality.html
functions = [
    ("betweenness", nx.betweenness_centrality),
    ("degree", nx.degree_centrality),
    ("closeness", nx.closeness_centrality),
    ("load_centrality", nx.load_centrality),
    ("eigenvector", nx.eigenvector_centrality),
    ("harmonic", nx.harmonic_centrality),
    ("subgraph", nx.subgraph_centrality)
]


def create_dataset(g=None, path=None):
    if g is None:
        g = load_graph(path)
    df = pd.DataFrame()
    assert g
    g_temp = g.copy()
    for func in functions:
        name = func[0]
        metric = func[1](g)
        for key, value in metric.items():
            metric[key] = {name: value}
        nx.set_node_attributes(g_temp, metric)

    for node in g_temp.nodes:
        node_to_add = g_temp.nodes[node]
        node_to_add['node'] = node
        df = df.append(node_to_add, ignore_index=True)
    df = df.set_index("node", drop=True)
    df["Y_true"] = df["Y_true"].astype("category")
    return df, g_temp


def split_dataset(df):
    train_df = df.drop(columns=["Y_true"])
    y = df["Y_true"]
    train_set, test_set, y_train, y_test = train_test_split(
        train_df,
        y,
        test_size=0.8,
        stratify=y,
        random_state=seed,
    )
    return train_set, test_set, y_train, y_test
