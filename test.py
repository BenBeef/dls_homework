# coding:utf-8

from typing import List

import numpy as np


class Value:

    def __init__(self, name, *inputs):
        self.name = name
        self.inputs = inputs

    def __repr__(self):
        return f"{self.name}->args:{[_input.name for _input in self.inputs]}"

def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    ### BEGIN YOUR SOLUTION
    if len(node_list) == 0:
        return node_list
    result: List[Value] = []
    visited = {}
    for node in node_list:
        if visited.get(node) is not None:
            continue
        topo_order = []
        topo_sort_dfs(node, visited, topo_order)
        result += topo_order
    return result
    ### END YOUR SOLUTION


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    ### BEGIN YOUR SOLUTION
    for _input in node.inputs:
        if visited.get(_input):
            continue
        topo_sort_dfs(_input, visited, topo_order)
    visited[node] = 1
    topo_order.append(node)


x1 = Value("x1")
x2 = Value("x2")
v1 = Value("v1", x1)
v2 = Value("v2", x2)
v3 = Value("v3", v1)
v4 = Value("v4", v1, v2)
v5 = Value("v5",  v2)
v6 = Value("v6",  v3, v4)
v7 = Value("v7",  v6, v5)

arr = [v7]

new_list = find_topo_sort(arr)
print(new_list)

