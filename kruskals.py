def add_edge(added_nodes, edge):
    node_zero_tree = -1
    node_one_tree = -1
    for tree, tree_index in zip(added_nodes, range(len(added_nodes))):
        for tree_edge in tree:
            if edge[0] in tree_edge:
                node_zero_tree = tree_index
            if edge[1] in tree_edge:
                node_one_tree = tree_index
    # print("tree1: {}, tree2: {}".format(node_zero_tree, node_one_tree))
    #   If none of the nodes have been added
    if node_zero_tree == node_one_tree and node_one_tree == -1:
        added_nodes.append([edge])
    #   If only the second node has been added
    elif node_zero_tree == -1:
        added_nodes[node_one_tree].append(edge)
    #   If only the first node has been added
    elif node_one_tree == -1:
        added_nodes[node_zero_tree].append(edge)
    #   If nodes belong to different trees
    elif node_zero_tree != node_one_tree:
        # Pop the later tree (so the index of the saved one does not change)
        if node_zero_tree < node_one_tree:
            added_nodes[node_zero_tree] = added_nodes[node_zero_tree] + added_nodes.pop(node_one_tree)
            added_nodes[node_zero_tree].append(edge)
        else:
            added_nodes[node_one_tree] = added_nodes[node_one_tree] + added_nodes.pop(node_zero_tree)
            added_nodes[node_one_tree].append(edge)
    #   Both nodes already belong to the same tree -> Do not add edge


#   Returns a maximum spanning tree (list of edges)
def kruskals(information_edges):
    #   Sort the edges by maximum weight
    information_edges = sorted(information_edges, key=lambda x: -x[2])
    added_nodes = [[information_edges.pop(0)]]
    for edge in information_edges:
        add_edge(added_nodes, edge)
    if len(added_nodes) > 1:
        print("Error: More than one resulting tree")
    return added_nodes[0]
