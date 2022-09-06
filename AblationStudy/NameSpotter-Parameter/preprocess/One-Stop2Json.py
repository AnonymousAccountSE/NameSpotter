from preprocess_method_name_parameter_pos import output2json_1
from preprocess_parameter import  preprocess_parameter_by_list
from split import split
from POS import pos
method_names = []
method_parameters = []
labels = ""
raw_file = "E:\\IdentifierQuality\\BadNames\\DataLabeling\\4504\\4504_revised.txt"
f_raw_file = open(raw_file,'r',encoding='utf-8')
raw_file_lines = f_raw_file.readlines()
for line in raw_file_lines:
    method_name = line.split(":")[0]
    method_parameter = line.split(":")[2]
    method_names.append(method_name)
    method_parameters.append(method_parameter)
    labels = labels + line.split(":")[5]

# preprocess word
output_word = "E:\\IdentifierQuality\\BadNames\\DataLabeling\\4504\\4504_revised_word.txt"
split(method_names,output_word)


# preprocess tag
output_tag = "E:\\IdentifierQuality\\BadNames\\DataLabeling\\4504\\4504_revised_tag.txt"
pos(output_word,output_tag)
# preprocess parameter
output_param = "E:\\IdentifierQuality\\BadNames\\DataLabeling\\4504\\4504_revised_param.txt"
preprocess_parameter_by_list(method_parameters, output_param)

output_labels = "E:\\IdentifierQuality\\BadNames\\DataLabeling\\4504\\4504_revised_label.txt"
write_label = open(output_labels,'w')
write_label.write(labels)

output_json = "E:\\IdentifierQuality\\BadNames\\DataLabeling\\4504\\method_name_tag_param.json"
output2json_1(output_word,output_tag,output_param, output_labels, output_json)