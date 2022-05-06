from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import pandas as pd
from lazy import lazy
from sklearn_hierarchical_classification.constants import ROOT


@dataclass(frozen=True)
class Data:
    dir_path: Path

    @lazy
    def name(self) -> str:
        return self.dir_path.stem

    @lazy
    def records(self) -> pd.DataFrame:
        path = self.dir_path.joinpath(self.name)
        return self._load_records(path)

    @lazy
    def records_train(self) -> pd.DataFrame:
        path = self.dir_path.joinpath(f"{self.name}_train")
        return self._load_records(path)

    @lazy
    def records_test(self) -> pd.DataFrame:
        path = self.dir_path.joinpath(f"{self.name}_test")
        return self._load_records(path)

    @lazy
    def hierarchy(self) -> nx.DiGraph:
        path = self.dir_path.joinpath(f"{self.name}.hf")
        return self._load_hierachy(path)

    @lazy
    def hierarchy_train(self) -> nx.DiGraph:
        path = self.dir_path.joinpath(f"{self.name}_train.hf")
        return self._load_hierachy(path)

    @lazy
    def hierarchy_test(self) -> nx.DiGraph:
        path = self.dir_path.joinpath(f"{self.name}_test.hf")
        return self._load_hierachy(path)

    @lazy
    def hierarchy_as_dict(self) -> Dict[str, Tuple[str]]:
        return self._hierarchy_to_dict(self.hierarchy)

    @lazy
    def hierarchy_train_as_dict(self) -> Dict[str, Tuple[str]]:
        return self._hierarchy_to_dict(self.hierarchy_train)

    @lazy
    def hierarchy_test_as_dict(self) -> Dict[str, Tuple[str]]:
        return self._hierarchy_to_dict(self.hierarchy_test)

    def _load_records(self, path: Path) -> pd.DataFrame:
        with path.open() as file:
            lines = file.read().split('\n')
            lines = [line.strip() for line in lines if line]

            rows = []

            for line in lines:
                cells = line.split(" ")

                features = {
                    index + 1: float(feature.split(":")[1])
                    for index, feature
                    in enumerate(cells[1:])
                }

                row = {
                    "labels": tuple(cells[0].split(",")),
                    **features
                }

                rows.append(row)

            return pd.DataFrame(rows)

    def _load_hierachy(self, path: Path) -> nx.Graph:
        hierarchy = nx.DiGraph()

        with path.open() as file:
            lines = file.read().split('\n')
            lines = [line.strip() for line in lines if line]

            edges = [line.split(" ") for line in lines]

            hierarchy.add_edges_from(edges)
            root = self._get_root(hierarchy)

            hierarchy = nx.relabel_nodes(hierarchy,{
                root:ROOT
            })

            return hierarchy

    def _get_root(self, hierarchy: nx.DiGraph) -> str:
        root = next(
            node
            for node, in_degree
            in hierarchy.in_degree()
            if in_degree == 0
        )

        return root

    def _hierarchy_to_dict(self, hierarchy: nx.DiGraph) -> Dict[str, Tuple[str]]:
        as_dict = {}

        for node in hierarchy.nodes:
            if node is ROOT:
                print('node is root',node)
                print('childs',hierarchy.out_edges(node))
                continue

            childs = [child for _, child in hierarchy.out_edges(node)]
            print(f"childs: of {node}",childs)
            if not childs:
                continue

            as_dict[node] = childs

        return as_dict
