import libpgm
import json
import data_prep


def read_dataset(file_path):
    dataset = []
    with open(file_path, 'r') as file:
        for line in file:
            attribute_values = line.split(',')
            #   Remove "\n" at the last attribute
            attribute_values[len(attribute_values)-1] = attribute_values[len(attribute_values)-1][:2]
            dataset.append(attribute_values)
    return dataset


def define_real_values(dataset_key, dataset):
    if dataset_key == "KDD+":
        for data_row in dataset:
            #   Remove the last (weird) attribute in each row
            data_row.pop(len(data_row)-1)
            data_row[0] = float(data_row[0])
            for index in range(4, 41):
                data_row[index] = float(data_row[index])
    else:
        print("There is not matching dataset key for: {}".format(dataset_key))


def calculate_intervals(dataset_key, dataset):
    if dataset_key == "KDD+":
        class_index = 41
        attr_values = {0: []}
        for index in range(4, 41):
            attr_values[index] = []

        for data_row in dataset:
            attr_values[0].append([data_row[0], data_row[class_index]])
            for index in range(4, 41):
                attr_values[index].append([data_row[index], data_row[class_index]])

        interval_cuts = {}
        for index in attr_values:
            interval_cuts[index] = data_prep.mdlp(attr_values[index])
            print(interval_cuts[index])


def main():
    dataset = read_dataset("res/NSL-KDD/KDDTrain+.txt")
    dataset_key = "KDD+"
    define_real_values(dataset_key, dataset)
    intervals = calculate_intervals(dataset_key, dataset)
    # print(dataset)


if __name__ == '__main__':
    main()
