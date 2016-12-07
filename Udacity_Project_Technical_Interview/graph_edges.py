def question3(G):
    """This script find the minimum \
    spanning tree that all vertices in a graph with the \
    smallest possible weight of edges."""

    if len(G) < 1:
        return G
    nodes = set(G.keys())
    min_span_tree = {}
    origin = G.keys()[0]
    min_span_tree[origin] = []

    while len(min_span_tree.keys()) < len(nodes):
        low_weight = float('inf')
        low_edge = None
        for node in min_span_tree.keys():
            edges = [(weight, vertex) for (vertex, weight) in G[node] if vertex not in min_span_tree.keys()]
            if len(edges) > 0:
                w, v = min(edges)
                if w < low_weight:
                    low_weight = w
                    low_edge = (node, v)
        min_span_tree[low_edge[0]].append((low_edge[1], low_weight))
        min_span_tree[low_edge[1]] = [(low_edge[0], low_weight)]
    return min_span_tree

print 'Test Cases for Question 3:'

print question3({})
# Should print {}

print question3({'A': []})
# Should print {'A': []}

print question3({'A': [('B', 2)], 'B': [('A', 2), ('C', 5)], 'C': [('B', 5)]})
#Shoud print {'A': [('B', 2)],
# 'B': [('A', 2), ('C', 5)], 
# 'C': [('B', 5)]}

print question3({'A': [('B', 3), ('E', 1)],
                 'B': [('A', 3), ('C', 9), ('D', 2), ('E', 2)],
                 'C': [('B', 9), ('D', 3), ('E', 7)],
                 'D': [('B', 2), ('C', 3)],
                 'E': [('A', 1), ('B', 2), ('C', 7)]})
# Should print
# {'A': [('E', 1)],
#  'C': [('D', 3)],
#  'B': [('E', 2), ('D', 2)],
#  'E': [('A', 1), ('B', 2)],
#  'D': [('B', 2), ('C', 3)]}