import data_prep


def read_dataset(dataset_key):
    if dataset_key == "KDD_train+" or dataset_key == "KDD_test+":
        if dataset_key == "KDD_train+":
            file_path = "res/mod_NSL/KDDTrain+.txt"
        else:
            file_path = "res/mod_NSL/KDDTest-21.txt"
        dataset = []
        attr_names = [
            'duration',
            'protocol_type',
            'service',
            'flag',
            'src_bytes',
            'dst_bytes',
            'land',
            'wrong_fragment',
            'urgent',
            'hot',
            'num_failed_logins',
            'logged_in',
            'num_compromised',
            'root_shell',
            'su_attempted',
            'num_root',
            'num_file_creations',
            'num_shells',
            'num_access_files',
            'num_outbound_cmds',
            'is_host_login',
            'is_guest_login',
            'count',
            'srv_count',
            'serror_rate',
            'srv_serror_rate',
            'rerror_rate',
            'srv_rerror_rate',
            'same_srv_rate',
            'diff_srv_rate',
            'srv_diff_host_rate',
            'dst_host_count',
            'dst_host_srv_count',
            'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate',
            'dst_host_srv_serror_rate',
            'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate',
            'class'
        ]
        value_space_dict = {'protocol_type': ['tcp', 'udp', 'icmp'],
                            'service': ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard',
                                        'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp',
                                        'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443',
                                        'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link',
                                        'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat',
                                        'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer',
                                        'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh',
                                        'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i',
                                        'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50'],
                            'flag': ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH'],
                            'class': []}
        with open(file_path, 'r') as file:
            for line in file:
                attribute_values = line.split(',')
                #   Remove "\n" at the last attribute
                attribute_values[len(attribute_values) - 1] = attribute_values[len(attribute_values) - 1][:len(attribute_values[len(attribute_values) - 1]) - 1]
                #   Remove the last (weird) attribute in each row
                attribute_values.pop(len(attribute_values) - 1)
                attribute_values[0] = float(attribute_values[0])
                for index in range(4, 41):
                    attribute_values[index] = float(attribute_values[index])
                data_row = dict(zip(attr_names, attribute_values))
                if data_row['class'] not in value_space_dict['class']:
                    value_space_dict['class'].append(data_row['class'])
                dataset.append(data_row)
        return dataset, value_space_dict
    elif dataset_key == "iris_train" or dataset_key == "iris_test":
        if dataset_key == "iris_train":
            file_path = "res/iris_data/iris_train.data"
        else:
            file_path = "res/iris_data/iris_test.data"
        attr_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
        dataset = []
        value_space_dict = {'class' : []}
        with open(file_path, 'r') as file:
            for line in file:
                if len(line) < 4:
                    continue
                attribute_values = line.split(',')
                #   Remove "\n" at the last attribute
                attribute_values[len(attribute_values) - 1] = attribute_values[len(attribute_values) - 1][:len(attribute_values[len(attribute_values) - 1]) - 1]
                for index in range(0, 4):
                    attribute_values[index] = float(attribute_values[index])
                data_row = dict(zip(attr_names, attribute_values))
                if data_row['class'] not in value_space_dict['class']:
                    value_space_dict['class'].append(data_row['class'])
                dataset.append(data_row)
        return dataset, value_space_dict
    elif dataset_key == "NB15_train" or dataset_key == "NB15_test":
        if dataset_key == "NB15_train":
            file_path = "res/UNSW-NB15/UNSW_NB15_training-set.csv"
        else:
            file_path = "res/UNSW-NB15/UNSW_NB15_testing-set.csv"
        attr_names = ["dur","proto","service","state","spkts","dpkts","sbytes","dbytes","rate","sttl","dttl","sload","dload","sloss","dloss","sinpkt","dinpkt","sjit","djit","swin","stcpb","dtcpb","dwin","tcprtt","synack","ackdat","smean","dmean","trans_depth","response_body_len","ct_srv_src","ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm","ct_dst_sport_ltm","ct_dst_src_ltm","is_ftp_login","ct_ftp_cmd","ct_flw_http_mthd","ct_src_ltm","ct_srv_dst","is_sm_ips_ports","class"]
        dataset = []
        value_space_dict = {'class' : []}
        line_num = 0
        with open(file_path, 'r') as file:
            for line in file:
                if line_num == 0:
                    line_num += 1
                    continue
                if len(line) < 4:
                    continue
                attribute_values = line.split(',')
                #   Remove the "id" attribute and the label
                attribute_values.pop(0)
                attribute_values.pop(len(attribute_values)-1)
                #   Remove "\n" at the last attribute
                attribute_values[len(attribute_values) - 1] = attribute_values[len(attribute_values) - 1][:len(attribute_values[len(attribute_values) - 1]) - 1]
                #   Make all the numbers to floats
                attribute_values[0] = float(attribute_values[0])
                for index in range(4, len(attribute_values)-2):
                    attribute_values[index] = float(attribute_values[index])
                data_row = dict(zip(attr_names, attribute_values))
                if data_row['class'] not in value_space_dict['class']:
                    value_space_dict['class'].append(data_row['class'])
                dataset.append(data_row)
        return dataset, value_space_dict


def calculate_intervals(dataset_key, dataset, value_space_dict):
    if dataset_key == "KDD_train+" or dataset_key == "NB15_train":
        print("Interval calculations has begun.")
        float_attr = {}
        #   Initiate lists for float attributes
        for attr in dataset[0]:
            if isinstance(dataset[0][attr], float):
                float_attr[attr] = []
        print("Floats initiated.")
        #   Parse all float values to lists
        for data_row in dataset:
            for attr in float_attr:
                float_attr[attr].append([data_row[attr], data_row['class']])
        print("Floats parsed to lists")
        #   Calculate the interval cuts
        interval_cuts = {}
        count_finished_attr = 0
        for attr in float_attr:
            interval_cuts[attr] = data_prep.mdlp(float_attr[attr])
            value_space_dict[attr] = interval_cuts[attr]
            count_finished_attr += 1
            print("Calculate intervals status: {} of {} attribute intervals created.".format(count_finished_attr, len(float_attr)))
        return interval_cuts
    else:
        print("Error: Unkown dataset key {} for interval calculation.".format(dataset_key))


def find_interval_value(data_value, interval_cuts):
    for cut in interval_cuts:
        if data_value < cut:
            return str(cut)
    return str(-1)


def discretize_to_intervals(dataset, intervals):
    for data_row in dataset:
        for attr_name in intervals:
            data_row[attr_name] = find_interval_value(data_row[attr_name], intervals[attr_name])