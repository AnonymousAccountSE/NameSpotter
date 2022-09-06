
def preprocess_parameter(input,output):
    with open(input, 'r', encoding='utf-8') as f:
    # with open("E:\IdentifierQuality\BadNames\DataLabeling\parameters.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        names = ""
        for line in lines:
            print(line.strip())
            if line.strip() == "null":
                names = names + "null\n"
                continue
            split_array_1 = line.strip().split(",")
            if len(split_array_1) == 1:
                split_array_2 = split_array_1[0].split(" ")
                names = names + split_array_2[-1] + "\n"
            else:
                for param in split_array_1:
                    split_array_2 = param.strip().split(" ")

                    names = names + split_array_2[-1] + " "
                names = names + "\n"
        print(names)

    # with open("E:\IdentifierQuality\BadNames\DataLabeling\parameters_nameOnly.txt", 'w') as write:
    with open(output, 'w') as write:
        write.write(names)


def preprocess_parameter_by_list(input,output):
    # with open("E:\IdentifierQuality\BadNames\DataLabeling\parameters.txt", 'r', encoding='utf-8') as f:
    lines = input
    names = ""
    for line in lines:
        print(line.strip())
        if line.strip() == "null":
            names = names + "null\n"
            continue
        split_array_1 = line.strip().split(",")
        if len(split_array_1) == 1:
            split_array_2 = split_array_1[0].split(" ")
            names = names + split_array_2[-1] + "\n"
        else:
            for param in split_array_1:
                split_array_2 = param.strip().split(" ")

                names = names + split_array_2[-1] + " "
            names = names + "\n"
    print(names)

    # with open("E:\IdentifierQuality\BadNames\DataLabeling\parameters_nameOnly.txt", 'w') as write:
    with open(output, 'w') as write:
        write.write(names)