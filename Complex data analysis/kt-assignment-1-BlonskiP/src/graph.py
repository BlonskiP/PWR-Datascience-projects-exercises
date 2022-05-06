import networkx as nx
import pandas as pd
import pickle as pkl
import os

COMMUNICATION_CSV = r"../data/communication.csv"
REPORTS_TO_CSV = r"../data/reportsto.csv"
directors = [86, 7, 27, 36, 69, 70, 85, 104, 121, 148, 156, 163]
managers = [76, 90, 136, 137, 143, 152, 47, 162]
FILFILENAME = "graph.pkl"

def load_data():
    communication = pd.read_csv(COMMUNICATION_CSV, sep=";")
    reportsto = pd.read_csv(REPORTS_TO_CSV, sep=";")

    reportsto["ID"] = reportsto["ID"].astype(int)
    reportsto = reportsto[reportsto.ReportsToID.apply(lambda x: x.isnumeric())]
    reportsto["ReportsToID"] = reportsto["ReportsToID"].astype(int)
    reportsto.reset_index(drop=True, inplace=True)

    communication["Sender"] = communication["Sender"].astype(int)
    communication["Recipient"] = communication["Recipient"].astype(int)

    return communication, reportsto


def create_edges(df: pd.DataFrame):
    """
    :param df: social system df. If has Sender column will work on communication df, else reportsTo df.
    :return: list of edges
    """
    edges = []
    if 'Sender' in df:
        unique_edges = df.groupby(by=["Sender", "Recipient"])  # without duplicates
        for pair, row in unique_edges:
            node_from = pair[0]
            node_to = pair[1]
            attribute = {"test": 0}
            if node_from is not node_to:
                edge = (node_from, node_to, attribute)
                edges.append(edge)
    else:
        employee_hierarchy = df.values.tolist()
        for hierarchy in employee_hierarchy:
            # employee raports to higher employee in hierarchy
            employee = hierarchy[0]
            higher_employee = hierarchy[1]
            attribute = {"test":0}
            edge = (employee,higher_employee,attribute)
            if employee is not higher_employee:
                edges.append(edge)
    return edges


def create_graph(save=False):
    communication, report = load_data()

    g = nx.Graph()
    # Create nodes - Sender is person in a network
    # Recipient is also a node, it is possible for someone not to write messages to anybody else.
    g.add_nodes_from(communication["Sender"].values.tolist())
    g.add_nodes_from(communication["Recipient"].values.tolist())

    communication_edges = create_edges(communication)
    hierarchy_edges = create_edges(report)
    g.add_edges_from(communication_edges)
    g.add_edges_from(hierarchy_edges)
    #setting classes
    # 0 - normal employee
    # 1 - manager
    # 2 - director
    classes = {}
    for node in g.nodes:
        classes[node] = {"Y_true":0}
        if node in managers:
            classes[node] = {"Y_true":1}
        elif node in directors:
            classes[node] = {"Y_true":2}
    nx.set_node_attributes(g,classes)
    if save:
        with open(FILFILENAME,"wb") as f:
            pkl.dump(g,f)

    return g

def load_graph(path=None):
        if path:
            file_path = path
        else:
            file_path = FILFILENAME
        with open (file_path,"rb") as f:
            g = pkl.load(f)
            return g

