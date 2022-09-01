
file = "../result/result_4504.txt"
output_bad_names = "../result/4504/bad_names.txt"
output_good_names = "../result/4504/good_names.txt"
output_true_positives = "../result/4504/true_positives.txt"
output_true_negatives = "../result/4504/true_negatives.txt"
output_false_negatives = "../result/4504/false_negatives.txt"
output_false_positives = "../result/4504/false_positives.txt"
bad_names = ""
good_names = ""
true_positives = ""
false_negatives = ""
true_negatives = ""
false_positives = ""
with open(file,'r') as f:
    lines = f.readlines()
    for line in lines:
        print(line)
        split_array = line.strip().split(":")
        name = split_array[0]
        # if len(split_array) <3:
        #     print(line)
        label = split_array[1]
        prediction = split_array[2]
        print(name)
        if label == '1':
            bad_names = bad_names + line.strip() + "\n"
            if prediction.strip() == '1':
                true_positives = true_positives +  name + "\n"
            else:
                false_negatives = false_negatives +  name + "\n"
        else:
            good_names = good_names + line.strip() + "\n"
            if prediction.strip() == '0':
                true_negatives = true_negatives +  name + "\n"
            else:
                false_positives = false_positives +  name + "\n"


# f1 = open(output_bad_names,'w')
# f1.write(bad_names)
#
f2 = open(output_true_positives,'w')
f2.write(true_positives)
#
f3 = open(output_false_negatives,'w')
f3.write(false_negatives)

f4 = open(output_good_names,'w')
f4.write(good_names)

f5 = open(output_true_negatives,'w')
f5.write(true_negatives)

f6 = open(output_false_positives,'w')
f6.write(false_positives)