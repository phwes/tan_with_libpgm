import math
import copy


#   TODO: Use binary search - not worth it
def cut_list(data_list, t):
    #   Boolean set if last iteration matched the value
    bol_last_val = False
    for i in range(len(data_list)):
        if bol_last_val:
            if data_list[i][0] != t:
                return data_list[:i], data_list[i:]
        elif data_list[i][0] == t:
            bol_last_val = True
    # print("Could not cut list: {} at value: {}".format(data_list, t))
    return data_list, []


#   Returns all classes present in given data set
def get_classes(data_set):
    k = []
    for data_point in data_set:
        if data_point[1] not in k:
            k.append(data_point[1])
    return k


#   Returns the proportion of a class in a list
def proportion(classes_list, data_list):
    count = {}
    for c in classes_list:
        count[c] = 0.0
    for data_point in data_list:
        count[data_point[1]] += 1.0
        # if data_point[1] == c:
        #     count += 1.0
    for c in count:
        if count[c] > 0.0:
            count[c] = count[c]/len(data_list)

    return count


#   Calculates the entropy of (classes in) a set S
def entropy(list_set, classes_list):
    # classes_list = get_classes(list_set)
    entropy_val = 0

    proportions_list = proportion(classes_list, list_set)
    for c in proportions_list:
        if proportions_list[c] > 0:
            prop = proportions_list[c]
            entropy_val -= prop*math.log2(prop)
    # for c in classes_list:
    #     prop = proportion(c, list_set)
    #     entropy_val -= prop*math.log2(prop)
    return entropy_val


def delta(entropy_s, entropy_one, entropy_two, k, k_one, k_two):
    return math.log2(math.pow(3, k) - 2) - (k * entropy_s - k_one * entropy_one - k_two * entropy_two)


def gain(entropy_s, entropy_one, entropy_two, s_one, s_two, n):
    return entropy_s - (len(s_one)/n) * entropy_one - (len(s_two)/n) * entropy_two


#   Decides whether or not to make a cut on this datapoint
def make_cut(s_list, cut_point):
    s_one, s_two = cut_list(s_list, cut_point[0])
    #   If list is empty, no cut was made
    if not s_two:
        return False
    one_class_list = get_classes(s_one)
    two_class_list = get_classes(s_two)
    #   s_class_list is the merge of one_class_list and two_class_list (optimization)
    s_class_list = copy.deepcopy(one_class_list)
    for c in two_class_list:
        if c not in s_class_list:
            s_class_list.append(c)
    entropy_s = entropy(s_list, s_class_list)
    entropy_one = entropy(s_one, one_class_list)
    entropy_two = entropy(s_two, two_class_list)
    k = len(s_class_list)
    k_one = len(one_class_list)
    k_two = len(two_class_list)
    n = len(s_list)
    delta_value = delta(entropy_s, entropy_one, entropy_two, k, k_one, k_two)
    gain_value = gain(entropy_s, entropy_one, entropy_two, s_one, s_two, n)
    right_hand_side = math.log2(n-1)/n + delta_value/n
    return gain_value > right_hand_side


#   data_list = [[1.0, "class_1"], [0.5, "class_4"], [23.6, "class_2"] ...]
#   Returns a list of cut sections [T1, T2, ...,Tn, -1] representing the partitions:
#   x <=T1, T1 < x <= T2 , ..., T(n-1) < x <= Tn, Tn < x
def mdlp(data_list):
    data_list.sort(key=lambda x: float(x[0]))
    s_list = copy.deepcopy(data_list)
    cuts = []
    last_val = None
    for data_point, i in zip(data_list, range(len(data_list))):
        if i % 10000 == 0:
            print("Data point {} out of {}".format(i, len(data_list)))
            print ("Num cuts: {}, Num left in list: {}".format(len(cuts), len(s_list)))
        #   No need to calculate the same cut again, nor can we make a cut on a list of length 1
        if data_point[0] == last_val or i == len(data_list)-1:
            continue
        #   Either make a cut, or add the data point to the upcoming cut
        if make_cut(s_list, data_point):
            cuts.append(data_point[0])
            #   New S partition only contains the remaining data points
            _, s_list = cut_list(s_list, data_point[0])
        last_val = data_point[0]

    #   Add the last partition that spans from last cut to infinity
    cuts.append(-1)
    return cuts
