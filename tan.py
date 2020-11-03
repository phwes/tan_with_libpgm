import json
import math
import os.path
import random


def save_probs(px, pxy, pxyc):
    px_file = open("saved_data/px.data", "w")
    json.dump(px, px_file)
    px_file.close()
    pxy_file = open("saved_data/pxy.data", "w")
    json.dump(pxy, pxy_file)
    pxy_file.close()
    pxyc_file = open("saved_data/pxyc.data", "w")
    json.dump(pxyc, pxyc_file)
    pxyc_file.close()


def load_probs():
    px_file = open("saved_data/px.data", "r")
    px = json.load(px_file)
    px_file.close()
    pxy_file = open("saved_data/pxy.data", "r")
    pxy = json.load(pxy_file)
    pxy_file.close()
    pxyc_file = open("saved_data/pxyc.data", "r")
    pxyc = json.load(pxyc_file)
    pxyc_file.close()
    return px, pxy, pxyc


#   Calculates the mutual information between two nodes (attributes)
def calc_mutual_information(classes, value_space_dict, dataset, information_edges):
    tot_sum = 0.0
    px = {}
    pxy = {}
    pxyc = {}
    count_x = {}
    count_xy = {}

    if os.path.exists("saved_data/pxyc.data"):
        px, pxy, pxyc = load_probs()
        print("Loaded probs from files")
    else:
        for c in classes:
            px[c] = {}
            pxy[c] = {}
            pxyc[c] = {}
            count_x[c] = {}
            count_xy[c] = {}

        #   Count specific classification and attribute values
        count_class = {}
        for data_row in dataset:
            c = data_row["class"]
            if c in count_class:
                count_class[c] += 1
            else:
                count_class[c] = 1
            for attr_name in data_row:
                if attr_name == "class":
                    continue
                else:
                    if attr_name in count_x[c].keys():
                        if data_row[attr_name] in count_x[c][attr_name].keys():
                            count_x[c][attr_name][data_row[attr_name]] += 1
                        else:
                            count_x[c][attr_name][data_row[attr_name]] = 1
                    else:
                        count_x[c][attr_name] = {}
                        count_x[c][attr_name][data_row[attr_name]] = 1

        #   Count combinations of attribute values
        for data_row in dataset:
            c = data_row["class"]
            for (attr_name_x, index_x) in zip(data_row, range(len(data_row))):
                if attr_name_x == "class":
                    continue
                for (attr_name_y, index_y) in zip(data_row, range(len(data_row))):
                    if attr_name_y == "class" or index_x > index_y:
                        continue
                    if attr_name_x in count_xy[c].keys():
                        if attr_name_y in count_xy[c][attr_name_x].keys():
                            if data_row[attr_name_x] in count_xy[c][attr_name_x][attr_name_y].keys():
                                if data_row[attr_name_y] in count_xy[c][attr_name_x][attr_name_y][data_row[attr_name_x]].keys():
                                    count_xy[c][attr_name_x][attr_name_y][data_row[attr_name_x]][data_row[attr_name_y]] += 1.0
                                else:
                                    count_xy[c][attr_name_x][attr_name_y][data_row[attr_name_x]][data_row[attr_name_y]] = 1.0
                            else:
                                count_xy[c][attr_name_x][attr_name_y][data_row[attr_name_x]] = {}
                                count_xy[c][attr_name_x][attr_name_y][data_row[attr_name_x]][data_row[attr_name_y]] = 1.0
                        else:
                            count_xy[c][attr_name_x][attr_name_y] = {}
                            count_xy[c][attr_name_x][attr_name_y][data_row[attr_name_x]] = {}
                            count_xy[c][attr_name_x][attr_name_y][data_row[attr_name_x]][data_row[attr_name_y]] = 1.0
                    else:
                        count_xy[c][attr_name_x] = {}
                        count_xy[c][attr_name_x][attr_name_y] = {}
                        count_xy[c][attr_name_x][attr_name_y][data_row[attr_name_x]] = {}
                        count_xy[c][attr_name_x][attr_name_y][data_row[attr_name_x]][data_row[attr_name_y]] = 1.0

        #   Calculate the probabilities for P(x|c)
        for c in count_x:
            px[c] = {}
            for attr_name in count_x[c]:
                px[c][attr_name] = {}
                count_sum = 0.0
                for attr_val in count_x[c][attr_name]:
                    count_sum += count_x[c][attr_name][attr_val]
                for attr_val in count_x[c][attr_name]:
                    px[c][attr_name][attr_val] = count_x[c][attr_name][attr_val] / count_sum
                    if px[c][attr_name][attr_val] > 1.0:
                        print("Error: P(x|c) > 1!")

        #   Calculate the probabilities for P(x,y|c) and P(x,y,c)
        for c in count_xy:
            pxy[c] = {}
            pxyc[c] = {}
            for attr_name_x in count_xy[c]:
                pxy[c][attr_name_x] = {}
                pxyc[c][attr_name_x] = {}
                for attr_name_y in count_xy[c][attr_name_x]:
                    pxy[c][attr_name_x][attr_name_y] = {}
                    pxyc[c][attr_name_x][attr_name_y] = {}
                    for attr_val_x in count_xy[c][attr_name_x][attr_name_y]:
                        pxy[c][attr_name_x][attr_name_y][attr_val_x] = {}
                        pxyc[c][attr_name_x][attr_name_y][attr_val_x] = {}
                        for attr_val_y in count_xy[c][attr_name_x][attr_name_y][attr_val_x]:
                            pxy[c][attr_name_x][attr_name_y][attr_val_x][attr_val_y] = count_xy[c][attr_name_x][attr_name_y][attr_val_x][attr_val_y]/count_class[c]
                            pxyc[c][attr_name_x][attr_name_y][attr_val_x][attr_val_y] = count_xy[c][attr_name_x][attr_name_y][attr_val_x][attr_val_y]/len(dataset)
        save_probs(px, pxy, pxyc)
        print ("Calculated and saved probs")

    #   Calculate the mutual information over each edge
    for edge in information_edges:
        attr_name_x = edge[0]
        attr_name_y = edge[1]
        mutual_information = 0.0
        for c in px:
            for attr_val_x in pxy[c][attr_name_x][attr_name_y]:
                for attr_val_y in pxy[c][attr_name_x][attr_name_y][attr_val_x]:
                    pxyc_val = pxyc[c][attr_name_x][attr_name_y][attr_val_x][attr_val_y]
                    pxy_val = pxy[c][attr_name_x][attr_name_y][attr_val_x][attr_val_y]
                    px_val = px[c][attr_name_x][attr_val_x]
                    py_val = px[c][attr_name_y][attr_val_y]
                    mutual_information += pxyc_val * math.log2(pxy_val/(px_val * py_val))
        edge.append(mutual_information)


#   Removes the edge weights from all the edges
def remove_weights(edges):
    for edge in edges:
        edge.pop(2)


#   Choose one random node and makes it root (every edge points out from that node). edge = [from_node, to_node]
def make_directed(edges):
    root_index = random.randint(0, len(edges)-1)
    next_node = [edges[root_index][0]]
    directed_tree = []
    edges_to_remove = []
    while edges:
        current_node = next_node.pop(0)
        for edge in edges:
            if edge[0] == current_node:
                directed_tree.append(edge)
                next_node.append(edge[1])
                if edge not in edges_to_remove:
                    edges_to_remove.append(edge)
            elif edge[1] == current_node:
                directed_tree.append([edge[1], edge[0]])
                next_node.append(edge[0])
                if edge not in edges_to_remove:
                    edges_to_remove.append(edge)
        for remove_edge in edges_to_remove:
            edges.remove(remove_edge)
        edges_to_remove = []
    return directed_tree


def add_c_node(edges, nodes):
    for node in nodes:
        if node != "class":
            edges.append(["class", node])




