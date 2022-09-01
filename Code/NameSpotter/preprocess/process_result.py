import numpy
best_acc = 0.0
best_f1 = 0.0
best_threshold_acc = 0.0
best_threshold_f1 = 0.0
for i in numpy.arange(0,10.0,0.1):
    file_path = "../result/result_torch_method_name_param_pos_tag_" + str(round(i,1))+".json"
    f = open(file_path)
    lines = f.readlines()
    acc = float(lines[0].split(",")[0])
    f1 = float(lines[0].split(",")[1])
    print(acc)
    print(f1)
    if acc > best_acc:
        best_acc = acc
        best_threshold_acc = round(float(i),1)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold_f1 = round(float(i),1)

print(best_acc)
print(best_threshold_acc)
print(best_f1)
print(best_threshold_f1)