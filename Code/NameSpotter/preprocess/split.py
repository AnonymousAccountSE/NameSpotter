from spiral import ronin

def split(input_file, output_file):
    write_file = open(output_file, mode='w')
    with open(input_file, 'r',
              encoding="UTF-8") as file:
        lines = [line for line in file]
        for line in lines:
            if line == "_\n":
                print("______")
                write_file.write(str(line).lower() + " ")
                continue
            # print(ronin.split(line))
            line_split = line.split("\t")
            name_tokens = ronin.split(line_split[0])
            param_tokens = ronin.split(line_split[1])
            labels = line_split[2].strip()
            for i in range(len(name_tokens)):
                if i != len(name_tokens) - 1:
                    write_file.write(str(name_tokens[i]).lower() + " ")
                else:
                    write_file.write(str(name_tokens[i]).lower() + "\t")
            for i in range(len(param_tokens)):
                if i != len(param_tokens) - 1:
                    write_file.write(str(param_tokens[i]).lower() + " ")
                else:
                    write_file.write(str(param_tokens[i]).lower() + "\t" + labels + "\n")

def split(lines, output_file):
    write_file = open(output_file, mode='w')
    # with open(input_file, 'r',
    #           encoding="UTF-8") as file:
    #     lines = [line for line in file]
    for line in lines:
        if line == "_\n":
            print("______")
            write_file.write(str(line).lower() + " ")
            continue
        # print(ronin.split(line))
        # line_split = line.split("\t")
        name_tokens = ronin.split(line)
        # param_tokens = ronin.split(line_split[1])
        # labels = line_split[2].strip()
        for i in range(len(name_tokens)):
            if i != len(name_tokens) - 1:
                write_file.write(str(name_tokens[i]).lower() + " ")
            else:
                write_file.write(str(name_tokens[i]).lower() + "\n")
        # for i in range(len(param_tokens)):
        #     if i != len(param_tokens) - 1:
        #         write_file.write(str(param_tokens[i]).lower() + " ")
        #     else:
        #         write_file.write(str(param_tokens[i]).lower() + "\t" + labels + "\n")
if __name__ == '__main__':
    input_file = "E:\IdentifierQuality\BadNames\DataLabeling\\NameAndScoreAndParameterAndTag_Binary.txt"
    output_file = "E:\IdentifierQuality\BadNames\DataLabeling\\NameAndScoreAndParameterAndTagSplit_Binary.txt"
    split(input_file,output_file)