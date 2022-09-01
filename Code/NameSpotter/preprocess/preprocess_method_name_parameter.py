import json



text_list = []
param_list = []
label_list = []
pos_list =  open("E:\\IdentifierQuality\\BadNames\\DataLabeling\\Names_POS.txt",'r').readlines()


with open("E:\\IdentifierQuality\\BadNames\\DataLabeling\\NameAndScoreAndParameterSplit_Binary.txt",'r') as f:
    lines = f.readlines()
    for line in lines:
        print(line.strip())
        line_array = line.split("\t")
        label_list.append(line_array[2].strip())
        text_list_each = line_array[0]
        param_list_each = line_array[1]
        text_list.append(text_list_each)
        param_list.append(param_list_each)

    for text in text_list:
        print(text)

    for param in param_list:
        print(param)

    for label in label_list:
        print(label)

    # line = ""
    # for index in range(0,len(text_list)):
    #     line = line + text_list[index]+","+label_list[index] + "\n"

    # with open("E:\\IdentifierQuality\\BadNames\\DataLabeling\\NameAndScoreSplit_comma.csv",'w') as write:
    #     write.write(line)
    #
    # print(len(text_list))
    # print(len(label_list))
    #
    medium_dic = {}
    label_dic = {}

    cnt_bad_name = 0
    cnt_good_name = 0

    for index, text_each in enumerate(text_list):
        if label_list[index] == "1":
            cnt_good_name = cnt_good_name + 1
        elif label_list[index] == "0":
            cnt_bad_name = cnt_bad_name + 1
        else:
            print("error!")
        if cnt_bad_name <= 562 and cnt_good_name <=562:
            inner_dic = {"text": text_each, "param":param_list[index], "pos": pos_list[index].strip(), "label": label_list[index]}
            medium_dic[str(index)] = inner_dic
        else:
            if cnt_bad_name <= 562 and label_list[index] == "0":
                inner_dic = {"text": text_each, "param": param_list[index], "pos": pos_list[index].strip(),
                             "label": label_list[index]}
                medium_dic[str(index)] = inner_dic
            if cnt_good_name <= 562 and label_list[index] == "1":
                inner_dic = {"text": text_each, "param": param_list[index], "pos": pos_list[index].strip(),
                             "label": label_list[index]}
                medium_dic[str(index)] = inner_dic
    # print(medium_dic)
    print(len(medium_dic))

    # output method name
    # token_set = set()
    # for key,value in medium_dic.items():
    #     token_set.update(value["text"].split(" "))
    # write_name = open("E:\IdentifierQuality\BadNames\DataLabeling\method_name_1124_split_balanced.txt",'w')
    # write_name.write(str(token_set))

    final_dic = {}
    train_dic = {}
    test_dic = {}
    for key,value in medium_dic.items():
        if int(key) <100:
            train_dic[key] = value
        else:
            test_dic[key] = value

    final_dic["train"] = train_dic
    final_dic["test"] = test_dic

    # print(final_dic)


with open("E:\\IdentifierQuality\\BadNames\\DataLabeling\\method_name_1124_param_pos_split_binary_balanced.json",'w') as f1:
    json.dump(final_dic,f1)


