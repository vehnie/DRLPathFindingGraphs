import networkx as nx
import matplotlib.pyplot as plt
from networkx.generators.random_graphs import fast_gnp_random_graph
import json
import random
import numpy as np

class CreationGraph:
    def __init__(self, predefined=False, digraph=False, size='small', n=10, p=0.5):
        self.__json_path = './graphs/graphs.json'
        self.__predefined = predefined
        self.__digraph = digraph

        if self.__predefined:
            predefined_graphs = self.__create_predefined_graph(self.__digraph)
            self.__G = next(iter(predefined_graphs[size].values()))
        else:
            self.__G = self.__create_random_graph(n, p, self.__digraph)

    def __create_predefined_graph(self, digraph: bool):
        with open(self.__json_path, 'r') as file:
            data = json.load(file)

        graphs = {}
        for size, graph_data in data["graphs"].items():
            graphs[size] = {}
            for graph_name, edges in graph_data.items():
                G = nx.DiGraph() if digraph else nx.Graph()
                # Add edges to the graph
                for edge_str in edges:
                    edge = tuple(map(int, edge_str.strip("()").split(",")))
                    G.add_edge(*edge)
                graphs[size][graph_name] = G
        return graphs

    def __create_random_graph(self, n, p, digraph: bool):
        # Create graph based on probability distribution
        G = fast_gnp_random_graph(n, p, seed=None, directed=digraph)

        # Ensure the graph is connected (only for undirected graphs)
        if not digraph:
            while not nx.is_connected(G):
                for node in range(1, n+1):
                    potential_edges = [(node, other_node) for other_node in range(n) if
                                       node != other_node and not G.has_edge(node, other_node)]
                    if potential_edges:
                        edge = potential_edges[0]
                        G.add_edge(*edge)
        return G
    

    def get_graph(self):
        """Return just the graph object"""
        return self.__G

    def get_nodes(self):
        """Return the graph and its start/end nodes"""
        return self.__G  # Just return the graph since we don't select nodes here anymore

    def draw_graph(self):
        plt.figure(figsize=(8, 6))
        nx.draw(self.__G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=15,
                arrows=True)
        plt.show()

class GraphFactory:
    @staticmethod
    def create_erdos_renyi(n, p):
        """Create an Erdos-Renyi directed graph."""
        if n is None or n < 2:
            raise ValueError("Number of nodes must be at least 2")
        if p is None or not (0 <= p <= 1):
            raise ValueError("Probability must be between 0 and 1")
            
        # Create base directed graph
        G = nx.DiGraph()
        
        # Add all nodes
        for i in range(n):
            G.add_node(i)
        
        # Add random directed edges
        for i in range(n):
            for j in range(n):
                if i != j and random.random() < p:
                    G.add_edge(i, j)
        
        # For n=2, ensure there's at least one path in each direction
        if n == 2:
            G.add_edge(0, 1)
            G.add_edge(1, 0)
            return G
        
        # For n>2, ensure strong connectivity by adding a directed cycle
        nodes = list(G.nodes())
        random.shuffle(nodes)  # Randomize the order
        for i in range(len(nodes)):
            G.add_edge(nodes[i], nodes[(i + 1) % n])
        
        # Calculate and set node positions using spring layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        # Convert positions to dictionary format
        nx.set_node_attributes(G, {node: {'x': float(coords[0]), 'y': float(coords[1])} 
                                 for node, coords in pos.items()}, 'pos')
        
        return G

    @staticmethod
    def create_barabasi_albert(n, m):
        """Create a Barabasi-Albert directed graph."""
        # Create undirected first, then convert to directed
        G = nx.barabasi_albert_graph(n=n, m=m)
        # Convert to directed by randomly orienting edges
        D = nx.DiGraph()
        D.add_nodes_from(G.nodes())
        for u, v in G.edges():
            # Randomly choose direction
            if random.random() < 0.5:
                D.add_edge(u, v)
            else:
                D.add_edge(v, u)
        return D

    @staticmethod
    def create_scale_free(n, alpha=0.41, beta=0.54, gamma=0.05):
        """Create a scale-free directed graph using Bollobás model."""
        try:
            # Create directed scale-free graph
            G = nx.scale_free_graph(n)
            return G
        except Exception as e:
            print(f"Error in create_scale_free: {str(e)}")
            # Fallback to powerlaw_cluster_graph if scale_free_graph fails
            G = nx.powerlaw_cluster_graph(n, 3, 0.5)
            D = nx.DiGraph()
            D.add_nodes_from(G.nodes())
            for u, v in G.edges():
                if random.random() < 0.5:
                    D.add_edge(u, v)
                else:
                    D.add_edge(v, u)
            return D

    @staticmethod
    def get_valid_start_end_nodes(G):
        """Get valid start and end nodes from the graph that have a path between them."""
        nodes = list(G.nodes())
        while True:
            start = random.choice(nodes)
            end = random.choice(nodes)
            if start != end and nx.has_path(G, start, end):
                return start, end

'''if __name__ == '__main__':
    graph = CreationGraph(predefined=True, digraph=True, size='small')
    print("Graph created successfully!")
    graph.draw_graph()
    print(nx.to_dict_of_lists(graph.get_graph()))'''
