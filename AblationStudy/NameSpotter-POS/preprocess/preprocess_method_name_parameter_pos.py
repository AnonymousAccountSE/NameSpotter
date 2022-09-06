import json

def output2json_1(word_input,pos_input , param_input, labels, output):

    pos_list = open(pos_input, 'r').readlines()
    # param_pos_list = open("E:\\IdentifierQuality\\BadNames\\DataLabeling\\parameters_nameOnly_POS.txt", 'r').readlines()
    text_list = open(word_input, 'r').readlines()
    param_list = open(param_input, 'r').readlines()
    label_list = open(labels, 'r').readlines()

    print(len(pos_list))
    print(len(text_list))
    print(len(param_list))

    medium_dic = {}

    for index, text_each in enumerate(text_list):
        inner_dic = {"text": text_each.strip(), "param": param_list[index].strip(), "pos": pos_list[index].strip(),"label":label_list[index].strip()}
        medium_dic[str(index)] = inner_dic
    print(medium_dic)

    final_dic = {}
    train_dic = {}
    test_dic = {}
    for key, value in medium_dic.items():
        if int(key) < 100:
            train_dic[key] = value
        else:
            test_dic[key] = value
        test_dic[key] = value

    final_dic["train"] = train_dic
    final_dic["test"] = test_dic

        # print(final_dic)

    # with open("E:\\IdentifierQuality\\BadNames\\DataLabeling\\method_name_param_pos_tag_split_binary.json", 'w') as f1:
    with open(output, 'w') as f1:
        json.dump(final_dic, f1)


def output2json():
    text_list = []
    label_list = []
    param_list = []
    pos_list = open("E:\\IdentifierQuality\\BadNames\\DataLabeling\\Names_POS.txt", 'r').readlines()
    # param_pos_list = open("E:\\IdentifierQuality\\BadNames\\DataLabeling\\parameters_nameOnly_POS.txt", 'r').readlines()

    print(len(pos_list))
    # print(len(param_pos_list))
    with open("E:\IdentifierQuality\BadNames\DataLabeling\\NameAndScoreAndParameterAndTagSplit_Binary.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            print(line.strip())
            line_array = line.split("\t")
            label_list.append(line_array[-1].strip())
            text_list_each = line_array[0]
            param_list_each = line_array[1]
            text_list.append(text_list_each)
            param_list.append(param_list_each)

        for text in text_list:
            print(text)

        for label in label_list:
            print(label)

        # line = ""
        # for index in range(0,len(text_list)):
        #     line = line + text_list[index]+","+label_list[index] + "\n"
        #
        # with open("E:\\IdentifierQuality\\BadNames\\DataLabeling\\NameAndScoreSplit_comma.csv",'w') as write:
        #     write.write(line)
        #
        print(len(text_list))
        print(len(label_list))

        medium_dic = {}
        label_dic = {}

        for index, text_each in enumerate(text_list):
            inner_dic = {"text": text_each, "param": param_list[index], "pos": pos_list[index].strip(),
                          "label": label_list[index]}
            medium_dic[str(index)] = inner_dic
        print(medium_dic)

        final_dic = {}
        train_dic = {}
        test_dic = {}
        for key, value in medium_dic.items():
            if int(key) < 100:
                train_dic[key] = value
            else:
                test_dic[key] = value

        final_dic["train"] = train_dic
        final_dic["test"] = test_dic

        # print(final_dic)

    # with open("E:\\IdentifierQuality\\BadNames\\DataLabeling\\method_name_param_pos_tag_split_binary.json", 'w') as f1:
    with open("E:\\IdentifierQuality\\BadNames\\DataLabeling\\method_name_param_pos_tag_split_binary.json", 'w') as f1:
        json.dump(final_dic, f1)



