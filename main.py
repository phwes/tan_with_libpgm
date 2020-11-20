import json
import read_data
import time
import tan
import os.path
import kruskals


#   Returns a list with all the nodes (attribute names)
def get_nodes(dataset):
    nodes = []
    for attr in dataset[0]:
        nodes.append(attr)
    return nodes


#   Returns a list of all possible edges (in format [[node_a, node_b], ...]]
def get_edges(nodes_list):
    edges = []
    for i in range(len(nodes_list)):
        if nodes_list[i] == "class":
            continue
        for j in range(i+1, len(nodes_list)):
            if nodes_list[j] == "class":
                continue
            edges.append([nodes_list[i], nodes_list[j]])
    return edges


def get_classes(dataset):
    classes = []
    for data_row in dataset:
        if data_row["class"] not in classes:
            classes.append(data_row["class"])
    return classes


def save_json_to_file(data, file_path):
    dump_file = open(file_path, "w")
    json.dump(data, dump_file)
    dump_file.close()


def load_json_from_file(file_path):
    dump_file = open(file_path, "r")
    return json.load(dump_file)


def run_tan():
    # train_dataset_key = "KDD_train+"
    # test_dataset_key = "KDD_test+"
    train_dataset_key = "NB15_train"
    test_dataset_key = "NB15_test"
    if os.path.exists("saved_data/dataset.data") and os.path.exists("saved_data/value_space.data") and os.path.exists("saved_data/intervals.data"):
        dataset = load_json_from_file("saved_data/dataset.data")
        value_space_dict = load_json_from_file("saved_data/value_space.data")
        intervals = load_json_from_file("saved_data/intervals.data")
        print("Dataset, value space and intervals loaded from files")
    else:
        dataset, value_space_dict = read_data.read_dataset(train_dataset_key)
        print("Training dataset loaded.")
        intervals = read_data.calculate_intervals(train_dataset_key, dataset, value_space_dict)
        save_json_to_file(intervals, "saved_data/intervals.data")
        print("Discrete intervals created.")
        read_data.discretize_to_intervals(dataset, intervals)
        print("Floats discretized to intervals.")

        save_json_to_file(value_space_dict, "saved_data/value_space.data")
        save_json_to_file(dataset, "saved_data/dataset.data")

    nodes = get_nodes(dataset)
    edges = get_edges(nodes)
    classes = get_classes(dataset)

    #   Create MST GraphSkeleton with Kruskal's algorithm
    information_edges = edges[:]
    tan.calc_mutual_information(classes, dataset, information_edges)
    print("Mutual information between edges completed.")
    # save_json_to_file(information_edges, "saved_data/information_edges.data")

    #   Get MST (Maximum spanning tree)
    mst = kruskals.kruskals(information_edges)
    print("Created MST.")
    tan.remove_weights(mst)
    print("Removed weights from MST.")
    tree, root_node, parent_of_dict = tan.make_directed(mst)
    print("Converted undirected tree to directed tree.")
    tan.add_c_node(tree, nodes)
    print("Added class node.")
    #   We now have the TAN structure (but no weights)

    #   Calculate the probs needed for prediction
    tan.calc_bayes_probs(dataset, parent_of_dict, root_node)
    print("Calculated last probabilities for prediction.")

    # for data_row in dataset:
    #     if data_row["class"] == "dos" and data_row["duration"] == "1.0":
    #     # if data_row["class"] == "dos":
    #         print(data_row)

    #   Read test dataset
    test_dataset, _ = read_data.read_dataset(test_dataset_key)
    read_data.discretize_to_intervals(test_dataset, intervals)

    #   Make prediction on test dataset
    tan.predict_dataset(test_dataset, root_node, classes)


def run_iris():
    train_dataset_key = "iris_train"
    test_dataset_key = "iris_test"
    dataset, value_space_dict = read_data.read_dataset(train_dataset_key)
    print("Training dataset loaded.")
    intervals = read_data.calculate_intervals("KDD_train+", dataset, value_space_dict)
    print("Discrete intervals created.")
    read_data.discretize_to_intervals(dataset, intervals)
    print("Floats discretized to intervals.")

    nodes = get_nodes(dataset)
    edges = get_edges(nodes)
    classes = get_classes(dataset)

    #   Create MST GraphSkeleton with Kruskal's algorithm
    information_edges = edges[:]
    tan.calc_mutual_information(classes, dataset, information_edges)
    print("Mutual information between edges completed.")
    # save_json_to_file(information_edges, "saved_data/information_edges.data")

    #   Get MST (Maximum spanning tree)
    mst = kruskals.kruskals(information_edges)
    print("Created MST.")
    tan.remove_weights(mst)
    print("Removed weights from MST.")
    tree, root_node, parent_of_dict = tan.make_directed(mst)
    print("Converted undirected tree to directed tree.")
    tan.add_c_node(tree, nodes)
    print("Added class node.")
    #   We now have the TAN structure (but no weights)

    #   Calculate the probs needed for prediction
    tan.calc_bayes_probs(dataset, parent_of_dict, root_node)
    print("Calculated last probabilities for prediction.")

    #   Read test dataset
    test_dataset, _ = read_data.read_dataset(test_dataset_key)
    read_data.discretize_to_intervals(test_dataset, intervals)

    #   Make prediction on test dataset
    tan.predict_dataset(test_dataset, root_node)


def main():
    print("Begin execution.")
    start_time = time.time()
    run_tan()
    # run_iris()
    end_time = time.time()
    tot_time = end_time - start_time
    print("Done executing. Total execution time: {}".format(tot_time))


if __name__ == '__main__':
    main()
