from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from copy import deepcopy


def get_data_matrix(dataset):
    data_matrix = []
    class_label_array = []
    real_feature_names = []
    for data_entry_name in dataset[0]:
        if isinstance(dataset[0][data_entry_name], float):
            real_feature_names.append(data_entry_name)
    print("Attribute names: {}".format(real_feature_names))
    for data_row in dataset:
        class_label_array.append(data_row['class'])
        matrix_row = []
        for data_entry_name in real_feature_names:
            matrix_row.append(data_row[data_entry_name])
        data_matrix.append(matrix_row)
    return data_matrix, class_label_array, real_feature_names

# def select_by_chi(data_matrix, class_label_array, feature_names):


def select_by_chi(dataset):
    data_matrix, class_label_array, feature_names = get_data_matrix(dataset)
    selector = SelectKBest(chi2, k=8)
    selector.fit(data_matrix, class_label_array)
    cols = selector.get_support(indices=True)
    print(cols)
    feature_subset = []
    for col in cols:
        feature_subset.append(feature_names[col])
    return feature_subset


def reduce_dataset(dataset, real_feature_names, value_space_dict):
    selected_attribute_names = []
    for attr_name in value_space_dict:
        selected_attribute_names.append(attr_name)
    selected_attribute_names += real_feature_names
    new_dataset = []
    for data_row in dataset:
        new_row = {}
        for attr_name in selected_attribute_names:
            new_row[attr_name] = data_row[attr_name]
        new_dataset.append(new_row)
    return new_dataset





# selected_features = select_by_chi([[1,3,4,5],[2,4,6,83],[5.3,23,7.3,7]],
#                     ["Normal", "Intrusion", "Intrusion"],
#                     ["count", "time", "bytes", "errors"])
# print(selected_features)