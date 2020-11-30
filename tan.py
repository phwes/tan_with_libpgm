import json
import math
import os.path
import random


def save_json_to_file(data, file_path):
    dump_file = open(file_path, "w")
    json.dump(data, dump_file)
    dump_file.close()


def load_json_from_file(file_path):
    dump_file = open(file_path, "r")
    return json.load(dump_file)


#   Calculates the mutual information between two nodes (attributes)
def calc_mutual_information(classes, dataset, information_edges):
    tot_sum = 0.0
    pc = {}
    px = {}
    pxy = {}
    pxyc = {}
    count_x = {}
    count_xy = {}

    if os.path.exists("saved_data/pxyc.data"):
        pc = load_json_from_file("saved_data/pc.data")
        px = load_json_from_file("saved_data/px.data")
        pxy = load_json_from_file("saved_data/pxy.data")
        pxyc = load_json_from_file("saved_data/pxyc.data")
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

        #   Calc P(c)
        for c in count_class:
            pc[c] = count_class[c] / len(dataset)

        #   Count combinations of attribute values
        for data_row in dataset:
            c = data_row["class"]
            for (attr_name_x, index_x) in zip(data_row, range(len(data_row))):
                if attr_name_x == "class":
                    continue
                for (attr_name_y, index_y) in zip(data_row, range(len(data_row))):
                    if attr_name_y == "class" or index_x >= index_y:
                        continue
                    if attr_name_x in count_xy[c].keys():
                        if attr_name_y in count_xy[c][attr_name_x].keys():
                            if data_row[attr_name_x] in count_xy[c][attr_name_x][attr_name_y].keys():
                                if data_row[attr_name_y] in count_xy[c][attr_name_x][attr_name_y][
                                    data_row[attr_name_x]].keys():
                                    count_xy[c][attr_name_x][attr_name_y][data_row[attr_name_x]][
                                        data_row[attr_name_y]] += 1.0
                                else:
                                    count_xy[c][attr_name_x][attr_name_y][data_row[attr_name_x]][
                                        data_row[attr_name_y]] = 1.0
                            else:
                                count_xy[c][attr_name_x][attr_name_y][data_row[attr_name_x]] = {}
                                count_xy[c][attr_name_x][attr_name_y][data_row[attr_name_x]][
                                    data_row[attr_name_y]] = 1.0
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
                            pxy[c][attr_name_x][attr_name_y][attr_val_x][attr_val_y] = \
                            count_xy[c][attr_name_x][attr_name_y][attr_val_x][attr_val_y] / count_class[c]
                            pxyc[c][attr_name_x][attr_name_y][attr_val_x][attr_val_y] = \
                            count_xy[c][attr_name_x][attr_name_y][attr_val_x][attr_val_y] / len(dataset)

        #   Calculate the prior estimate
        #   TODO: Fix this, some values are missed
        prior_estimates = {}
        count_attr_total = {}
        count_attr_val_total = {}
        for c in count_x:
            for attr_name in count_x[c]:
                if attr_name not in prior_estimates:
                    if attr_name not in count_attr_total:
                        count_attr_total[attr_name] = 0
                        count_attr_val_total[attr_name] = {}
                    for attr_val in count_x[c][attr_name]:
                        count_attr_total[attr_name] += count_x[c][attr_name][attr_val]
                        if attr_val not in count_attr_val_total[attr_name]:
                            count_attr_val_total[attr_name][attr_val] = count_x[c][attr_name][attr_val]
                        else:
                            count_attr_val_total[attr_name][attr_val] += count_x[c][attr_name][attr_val]
        for attr_name in count_attr_val_total:
            if attr_name not in prior_estimates:
                prior_estimates[attr_name] = {}
            for attr_val in count_attr_val_total[attr_name]:
                tot_val_tmp = count_attr_val_total[attr_name][attr_val]
                tot_attr_tmp = count_attr_total[attr_name]
                prior_estimates[attr_name][attr_val] = tot_val_tmp / tot_attr_tmp

        save_json_to_file(prior_estimates, "saved_data/prior_estimates.data")
        save_json_to_file(pc, "saved_data/pc.data")
        save_json_to_file(px, "saved_data/px.data")
        save_json_to_file(pxy, "saved_data/pxy.data")
        save_json_to_file(pxyc, "saved_data/pxyc.data")
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
                    mutual_information += pxyc_val * math.log2(pxy_val / (px_val * py_val))
        edge.append(mutual_information)


#   Removes the edge weights from all the edges
def remove_weights(edges):
    for edge in edges:
        edge.pop(2)


#   Choose one random node and makes it root (every edge points out from that node). edge = [from_node, to_node]
def make_directed(edges):
    root_index = random.randint(0, len(edges) - 1)
    root_node = edges[root_index][0]
    parent_of_dict = {}
    next_node = [root_node]
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
                if edge[1] not in parent_of_dict:
                    parent_of_dict[edge[1]] = edge[0]
            elif edge[1] == current_node:
                directed_tree.append([edge[1], edge[0]])
                next_node.append(edge[0])
                if edge not in edges_to_remove:
                    edges_to_remove.append(edge)
                if edge[0] not in parent_of_dict:
                    parent_of_dict[edge[0]] = edge[1]
        for remove_edge in edges_to_remove:
            edges.remove(remove_edge)
        edges_to_remove = []
        save_json_to_file(parent_of_dict, "saved_data/parent_of_dict.data")
    return directed_tree, root_node, parent_of_dict


def add_c_node(edges, nodes):
    for node in nodes:
        if node != "class":
            edges.append(["class", node])


def calc_bayes_probs(dataset, parent_of_dict, root_node):
    count_given_parent_c = {}
    px_given_parent_c = {}

    #   Count the occurrences with parent-child values
    for data_row in dataset:
        c = data_row["class"]
        if c not in count_given_parent_c:
            count_given_parent_c[c] = {}
        for child_name in data_row:
            if child_name == root_node or child_name == "class":
                continue
            child_value = data_row[child_name]
            parent_name = parent_of_dict[child_name]
            parent_value = data_row[parent_name]
            if parent_name not in count_given_parent_c[c]:
                count_given_parent_c[c][parent_name] = {}
            if child_name not in count_given_parent_c[c][parent_name]:
                count_given_parent_c[c][parent_name][child_name] = {}
            if parent_value not in count_given_parent_c[c][parent_name][child_name]:
                count_given_parent_c[c][parent_name][child_name][parent_value] = {}
            if child_value not in count_given_parent_c[c][parent_name][child_name][parent_value]:
                count_given_parent_c[c][parent_name][child_name][parent_value][child_value] = 1
            else:
                count_given_parent_c[c][parent_name][child_name][parent_value][child_value] += 1

    #   Calculate the probabilities P(x|c,Pa(x))
    for c in count_given_parent_c:
        for parent_name in count_given_parent_c[c]:
            for child_name in count_given_parent_c[c][parent_name]:
                for parent_value in count_given_parent_c[c][parent_name][child_name]:
                    child_sum = 0
                    for child_value in count_given_parent_c[c][parent_name][child_name][parent_value]:
                        child_sum += count_given_parent_c[c][parent_name][child_name][parent_value][child_value]
                    for child_value in count_given_parent_c[c][parent_name][child_name][parent_value]:
                        if c not in px_given_parent_c:
                            px_given_parent_c[c] = {}
                        if child_name not in px_given_parent_c[c]:
                            px_given_parent_c[c][child_name] = {}
                        if parent_value not in px_given_parent_c[c][child_name]:
                            px_given_parent_c[c][child_name][parent_value] = {}
                        px_given_parent_c[c][child_name][parent_value][child_value] = \
                            count_given_parent_c[c][parent_name][child_name][parent_value][child_value] / child_sum
    save_json_to_file(px_given_parent_c, "saved_data/px_given_parent_c.data")


def smooth_prediction(data_row, root_node, pc, px, px_given_parent_c, parent_of_dict, prior_estimates_dict, data_len):
    n_0 = 5  # Same value as in TAN report
    n = data_len
    prob_value = {}
    min_val = 0.0
    for c in px:
        prob_value[c] = 1
        for attr_name in data_row:
            if attr_name == 'class':
                continue
            elif attr_name == root_node:
                attr_value = data_row[attr_name]
                #   TODO: This should always exist
                if attr_value not in prior_estimates_dict[attr_name]:
                    prior_estimate = min_val
                else:
                    prior_estimate = prior_estimates_dict[attr_name][attr_value]
                prob_parents = pc[c]
                if attr_value not in px[c][attr_name]:
                    term_one = min_val
                else:
                    term_one = (n * prob_parents) / (n * prob_parents + n_0) * \
                           px[c][attr_name][attr_value]
                term_two = n_0 / (n * prob_parents + n_0) * prior_estimate
                sum = term_one + term_two
                prob_value[c] *= sum
            else:
                parent_name = parent_of_dict[attr_name]
                parent_value = data_row[parent_name]
                attr_value = data_row[attr_name]
                #   TODO: This should always exist
                if attr_value not in prior_estimates_dict[attr_name]:
                    prior_estimate = min_val
                else:
                    prior_estimate = prior_estimates_dict[attr_name][attr_value]
                if parent_value not in px[c][parent_name]:
                    prob_parents = min_val
                else:
                    prob_parents = px[c][parent_name][parent_value]*pc[c]
                if parent_value not in px_given_parent_c[c][attr_name]:
                    term_one = min_val
                elif attr_value not in px_given_parent_c[c][attr_name][parent_value]:
                    term_one = min_val
                else:
                    term_one = (n*prob_parents)/(n*prob_parents+n_0)*px_given_parent_c[c][attr_name][parent_value][attr_value]
                term_two = n_0/(n*prob_parents+n_0)*prior_estimate
                sum = term_one + term_two
                prob_value[c] *= sum
    max_prob = 0.0
    prediction = None
    for c in prob_value:
        if prob_value[c] > max_prob:
            max_prob = prob_value[c]
            prediction = c
    return prediction


#   Make a prediction on a single data row
def predict_data_row(data_row, root_node, pc, px, px_given_parent_c, parent_of_dict):
    minimum_prob = 0.0
    score_c = {}
    for c in px:
        if data_row[root_node] not in px[c][root_node]:
            score_c[c] = minimum_prob
        else:
            score_c[c] = pc[c] * px[c][root_node][data_row[root_node]]
        for child_name in data_row:
            if child_name == root_node or child_name == "class":
                continue
            else:
                parent_name = parent_of_dict[child_name]
                parent_value = data_row[parent_name]
                child_value = data_row[child_name]
                if parent_value not in px_given_parent_c[c][child_name]:
                    score_c[c] = score_c[c] * minimum_prob
                elif child_value not in px_given_parent_c[c][child_name][parent_value]:
                    score_c[c] = score_c[c] * minimum_prob
                else:
                    score_c[c] = score_c[c] * px_given_parent_c[c][child_name][parent_value][child_value]
    most_likely = [None, 0.0]
    for c in score_c:
        if score_c[c] > most_likely[1]:
            most_likely = [c, score_c[c]]
    return most_likely[0]


def predict_dataset(test_dataset, root_node, classes):
    pc = load_json_from_file("saved_data/pc.data")
    px = load_json_from_file("saved_data/px.data")
    px_given_parent_c = load_json_from_file("saved_data/px_given_parent_c.data")
    parent_of_dict = load_json_from_file("saved_data/parent_of_dict.data")
    prior_estimates = load_json_from_file("saved_data/prior_estimates.data")
    #   Dictionary with entries [num_correct, num_incorrect]
    count_correct_dict = {}
    tot_correct = 0
    tot_wrong = 0
    false_positives = 0

    for data_row in test_dataset:
        # prediction = predict_data_row(data_row, root_node, pc, px, px_given_parent_c, parent_of_dict)
        prediction = smooth_prediction(data_row, root_node, pc, px, px_given_parent_c, parent_of_dict, prior_estimates, len(test_dataset))
        print("Prediction: {}, correct answer: {}".format(prediction, data_row['class']))
        #   Add entry in dict, if not exists
        if data_row['class'] not in count_correct_dict:
            count_correct_dict[data_row['class']] = [0, 0]
        #   Count if correct or not
        if data_row['class'] == prediction:
            count_correct_dict[data_row['class']][0] += 1
            tot_correct += 1
        else:
            count_correct_dict[data_row['class']][1] += 1
            tot_wrong += 1
        #   Count if false positive
        if (data_row['class'] == "0" and prediction != "0") or (
                data_row['class'] == "Normal" and prediction != "Normal"):
            false_positives += 1

    #   Calculate the accuracy in each classification
    for c in count_correct_dict:
        num_correct = count_correct_dict[c][0]
        num_wrong = count_correct_dict[c][1]
        tot_num = num_correct + num_wrong
        accuracy = num_correct / tot_num
        count_correct_dict[c].append(accuracy)
        print("Classification: {}, Accuracy: {}, Total number of entries: {}".format(c, accuracy, tot_num))
        print ("Num_Correct: {}, Num_Miss: {}".format(num_correct, num_wrong))

    print("Number of false positives: {}".format(false_positives))
    print("Overall accuracy: {}, Tot_correct: {}, Tot_wrong: {}".format(tot_correct / len(test_dataset), tot_correct,
                                                                        tot_wrong))
    print("Tot number of entries: {}".format(len(test_dataset)))
