import networkx as nx
import pandas as pd
from typing import List
from sklearn.base import BaseEstimator

ACTUAL_CLASS = 'actual_class'
PREDICTED = 'predicted_class'
TRUE_CLASS = 'Y_true'
BETWEENNESS = 'betweenness'
NUM_CLASSES = 3

class IDA_classfier(BaseEstimator):
    def get_params(self,deep):
        dict =  {
            "G":self._g,
            "uncover_rate":self._uncover_rate,
            "attribute":self._nodes_attribute
        }
        return dict
    def set_params(self, **parameters):
        ##TODO
        return self

    def __init__(self, G: nx.Graph, uncover_rate: float, attribute: str):
        self._g = G
        self._uncover_rate = uncover_rate
        self._nodes_attribute = attribute
        self._bootstrapped_data = self.bootstrap()

    def bootstrap(self):
        nodes_count = len(self._g.nodes)
        uncover_nodes_count = int(self._uncover_rate * nodes_count)  # how many nodes do we uncover for algorith

        uncover_nodes_list = self.nodes_uncover(self._nodes_attribute, uncover_nodes_count, self._g)
        self.uncover(self._g, uncover_nodes_list)

        self.set_default_class_score(self._g)
        train_set, test_set, y_train, y_test = self.prepare_sets(self._g, uncover_nodes_list)

        bootstrapped_data = [self._g, train_set, test_set, y_train, y_test]

        return bootstrapped_data

    def fit(self, classifier, stop_iterations):
        g, train, test, y_train, y_test = self._bootstrapped_data

        classifier.fit(X=train, y=y_train)
        predicted_nodes = classifier.predict(test)
        nodes = test.index
        for node, pred_y in zip(nodes, predicted_nodes):
            nx.set_node_attributes(g, {node: {PREDICTED: int(pred_y)}})

        changes_iterator = 0
        MAX_CHANGES = 3
        it = 0
        while changes_iterator != MAX_CHANGES and it < stop_iterations:
            change = False
            sorted_nodes = sorted(nodes, key=lambda item: test.loc[node, self._nodes_attribute], reverse=True)
            for node in sorted_nodes:
                for src, dest in g.edges(node):
                    if src == node:
                        self.update_score(g, node, dest)
                    elif dest == node:
                        self.update_score(g, node, src)
                for i in range(NUM_CLASSES):
                    key = f'class_{i}_score'
                    test.loc[node, key] = g.nodes[node][key]
                pred = int(classifier.predict([test.loc[node]])[0])
                if pred != g.nodes[node][PREDICTED]:
                    change = True

                nx.set_node_attributes(g, {node:
                                               {PREDICTED: pred}
                                           })
                if change:
                    changes_iterator = 0
                else:
                    changes_iterator += 1
                it += 1

        return g, test

    def predict(self, x):
        y = []
        for node in x:
                node_from_graph = self._g.nodes[node]
                if PREDICTED in node_from_graph:
                    node_class = node_from_graph[PREDICTED]
                    y.append(node_class)
                else:
                    node_class = node_from_graph[TRUE_CLASS]
                    y.append(node_class)
        return y

    def prepare_sets(self ,G: nx.Graph, uncover_nodes_list: List[int]):
        train_set = pd.DataFrame()  # contains known nodes
        test_set = pd.DataFrame()  # contains unknown nodes
        for node in G.nodes:
            if node in uncover_nodes_list:
                train_set = train_set.append(G.nodes[node], ignore_index=True)
            else:
                test_set = test_set.append(G.nodes[node], ignore_index=True)

        train_set.set_index('node', drop=True, inplace=True)
        test_set.set_index('node', drop=True, inplace=True)

        y_train = train_set[TRUE_CLASS]
        y_test = test_set[TRUE_CLASS]
        to_drop = [TRUE_CLASS, ACTUAL_CLASS]

        train_set.drop(columns=to_drop, inplace=True, axis=1)
        test_set.drop(columns=[TRUE_CLASS], inplace=True, axis=1)

        return train_set, test_set, y_train, y_test

    def update_class(self, G, node, neighbour, classname):
        # If destination node (neighbour) has class:value update source node score of this class.
        if classname in G.nodes[neighbour]:
            neighbour_class = int(G.nodes[neighbour][classname])
            key = f'class_{neighbour_class}_score'
            if key in G.nodes[node]:  # if already has this key
                update = G.nodes[node][key]
            else:  # if it hasnt got this class score (just to be sure)
                update = 1
            nx.set_node_attributes(G, {node: {
                key: update
            }})

    def update_score(self,G: nx.Graph, source_node: int, dest_node: int):
        self.update_class(G, source_node, dest_node, ACTUAL_CLASS)
        self.update_class(G, source_node, dest_node, PREDICTED)

    def nodes_uncover(self,attribute: str, nodes_count: int, G: nx.Graph):
        nodes_sorted = sorted(G.nodes, key=lambda item: G.nodes[item][attribute], reverse=True)
        node_uncover_list = [node for node in nodes_sorted][:nodes_count]
        return node_uncover_list

    def uncover(self,G: nx.Graph, nodes_to_uncover: List[int]):
        for node in nodes_to_uncover:
            G.nodes[node][ACTUAL_CLASS] = G.nodes[node]['Y_true']

    def set_default_class_score(self,G):
        # at the beginning all scores (of 3 classes) are 0
        for node in G.nodes:
            template = {
                node: {
                    "class_0_score": 0,
                    "class_1_score": 0,
                    "class_2_score": 0
                }
            }
            nx.set_node_attributes(G, template)

        for node in G.nodes:
            for src, dest in G.edges(node):
                # for each source node update class score
                if node == src:
                    self.update_score(G, node, dest)