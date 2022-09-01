import json



text_list = []
label_list = []

with open("E:\IdentifierQuality\BadNames\DataLabeling\\NameAndScoreSplit.txt",'r') as f:
    lines = f.readlines()
    for line in lines:
        text_list_each = ""
        print(line.strip())
        line_array = line.split(" ")
        for i in range(0, len(line_array)):

            if i == len(line_array) -1:
                label_list.append(line_array[i].strip())
            elif i == len(line_array) -2:
                text_list_each = text_list_each + line_array[i]
            else:
                text_list_each = text_list_each + line_array[i] + " "
        text_list.append(text_list_each)

    for text in text_list:
        print(text)

    for label in label_list:
        print(label)

    line = ""
    for index in range(0,len(text_list)):
        line = line + text_list[index]+","+label_list[index] + "\n"

    with open("E:\\IdentifierQuality\\BadNames\\DataLabeling\\NameAndScoreSplit_comma.csv",'w') as write:
        write.write(line)
    #
    print(len(text_list))
    print(len(label_list))

    medium_dic = {}
    label_dic = {}

    for index, text_each in enumerate(text_list):
        inner_dic = {"text": text_each, "label": label_list[index]}
        medium_dic[str(index)] = inner_dic
    print(medium_dic)

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

    print(final_dic)


with open("E:\\IdentifierQuality\\BadNames\\DataLabeling\\method_name.json",'w') as f1:
    json.dump(final_dic,f1)


