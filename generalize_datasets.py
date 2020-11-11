

def generalize_datset(load_file_path, save_file_path, training_bool):
    new_dataset = []
    class_index = 41
    # path = "res/NSL-KDD/KDDTest+.txt"
    if training_bool:
        dos = ["back", "land", "neptune", "pod", "smurf", "teardrop"]
        probe = ["ipsweep", "nmap", "portsweep", "satan"]
        r2l = ["spy", "warezclient", "ftp_write", "guess_passwd", "imap", "multihop", "phf", "warezmaster"]
        u2r = ["buffer_overflow", "ps", "loadmodule", "rootkit", "perl"]
    else:
        dos = ["apache2", "back", "land", "mailbomb", "neptune", "pod", "smurf", "teardrop", "worm", "processtable", "udpstorm"]
        probe = ["ipsweep", "mscan", "nmap", "portsweep", "saint", "satan"]
        r2l = ["ftp_write", "guess_passwd", "httptunnel", "imap", "multihop", "named", "phf", "sendmail", "snmpgetattack", "snmpguess", "xlock", "warezmaster", "xsnoop"]
        u2r = ["buffer_overflow", "ps", "perl", "loadmodule", "sqlattack", "xterm", "rootkit"]
    with open(load_file_path, 'r') as file:
        for line in file:
            data_row = line.split(',')
            if data_row[class_index] in dos:
                data_row[class_index] = "dos"
            elif data_row[class_index] in probe:
                data_row[class_index] = "probe"
            elif data_row[class_index] in r2l:
                data_row[class_index] = "r2l"
            elif data_row[class_index] in u2r:
                data_row[class_index] = "u2r"
            elif data_row[class_index] == "normal":
                pass
            else:
                print("{} does not match any category.".format(data_row[class_index]))
            new_dataset.append(data_row)

    with open(save_file_path, 'w') as file:
        for data_row in new_dataset:
            text_row = ','.join(str(elem) for elem in data_row)
            # print(text_row)
            file.write(text_row)


generalize_datset("res/NSL-KDD/KDDTrain+.txt", "res/mod_NSL/KDDTrain+.txt", True)
