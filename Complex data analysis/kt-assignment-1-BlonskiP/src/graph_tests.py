from src.graph import *
from src.dataset_loader import *

def test_load_data():
    com, raport = load_data()
    assert not com.empty
    assert not raport.empty


def test_edges():
    com, raport = load_data()
    edges = create_edges(com)
    assert edges
    edges = create_edges(raport)
    assert edges
    pass


def test_graph_creation():
    g = create_graph(True)
    assert len(g.nodes)
    assert len(g.edges)


def test_dataset_creation():
    data = create_dataset()
    for f in functions:
        name = f[0]
        assert name in data

