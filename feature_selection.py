from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def select_by_chi(data_matrix, class_label_array, feature_names):
    selector = SelectKBest(chi2, k=2)
    selector.fit(data_matrix, class_label_array)
    cols = selector.get_support(indices=True)
    print(cols)
    feature_subset = []
    for col in cols:
        feature_subset.append(feature_names[col])
    return feature_subset


# selected_features = select_by_chi([[1,3,4,5],[2,4,6,83],[5.3,23,7.3,7]],
#                     ["Normal", "Intrusion", "Intrusion"],
#                     ["count", "time", "bytes", "errors"])
# print(selected_features)