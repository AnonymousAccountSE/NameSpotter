from xml.dom.minidom import Document

import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")
# nlp.disable_pipes('tok2vec', 'ner')
# print(nlp.pipe_names)

# print(nlp.get_pipe("tagger").labels)
'''
read from file
'''

def pos(input, output):
    write_file = open(output, mode='w')
    with open(input, 'r', encoding='utf-8') as f:

        # write_file = open("E:\\BadMethodName\\Evaluation\\Analysis_20\\FiveCategories_pos.txt", mode='w')
        # with open("E:\\BadMethodName\\Evaluation\\Analysis_20\\FiveCategories.txt",'r',encoding='utf-8') as f:

        cnt = 0
        for line in f:
            if cnt % 1000 == 0:
                print(cnt)
            # if cnt > 2086000:
            if cnt >= 0:
                doc = nlp(line.strip().replace(",", " "))
                # write_file.write(line)
                for token in doc:
                    # print((token.head.text, token.text, token.pos_, token.dep_, spacy.explain(token.dep_)))
                    # print(token.dep_, token.head.pos_,  token.pos_, token.head.text, token.text)
                    # write_file.write(token.dep_ + "," + token.head.pos_ + "," + token.pos_ + "," + token.head.text + "," + token.text+"\n")
                    write_file.write(token.pos_ + " ")
                    # print(token.pos_)
                    # write_file.write(token.dep_ + "," + token.head.pos_ + "," + token.pos_ +";")
                # write_file.write("----------------\n")
                write_file.write("\n")
                # write_file.write("\n----------------\n")
            cnt = cnt + 1

# write_file = open("E:\\BadMethodName\\icse2020\\TrainingData\\TrainingData_Filtered\\Patterns_parsedMethodNameTokens_public.txt",mode='w')
# write_file = open("E:\\IdentifierQuality\\Patterns_parsedMethodNameTokens_non_public.txt",mode='w')
#
# write_file = open("E:\\IdentifierQuality\\Patterns_parsedMethodNameTokens_public.txt",mode='w')
# write_file = open("E:\\IdentifierQuality\\Patterns_parsedMethodNameTokens_public.txt", mode='a')
#
# write_file = open("E:\\IdentifierQuality\\JDK1.8\\Patterns_parsedMethodNameTokens_non_public.txt", mode='a')
# write_file = open("E:\\IdentifierQuality\\JDK1.8\\Patterns_parsedMethodNameTokens_public.txt", mode='w')
#
#
#
# with open("E:\\BadMethodName\\icse2020\\TrainingData\\TrainingData_Filtered\\parsedMethodNameTokens.txt",'r') as f:
#
# with open("E:\\IdentifierQuality\\parsedMethodNameTokens_public.txt",'r',encoding='utf-8') as f:
# with open("E:\\IdentifierQuality\\parsedMethodNameTokens_non_public.txt",'r') as f:
#
# with open("E:\\IdentifierQuality\\JDK1.8\\parsedMethodNameTokens_non_public.txt",'r',encoding='utf-8') as f:
# with open("E:\\IdentifierQuality\\JDK1.8\\parsedMethodNameTokens_public.txt",'r',encoding='utf-8') as f:

# non-public

write_file = open("E:\\IdentifierQuality\\BadNames\\DataLabeling\\parameters_nameOnly_POS.txt", mode='w')
with open("E:\\IdentifierQuality\\BadNames\\DataLabeling\\parameters_nameOnly_split.txt",'r',encoding='utf-8') as f:

# write_file = open("E:\\BadMethodName\\Evaluation\\Analysis_20\\FiveCategories_pos.txt", mode='w')
# with open("E:\\BadMethodName\\Evaluation\\Analysis_20\\FiveCategories.txt",'r',encoding='utf-8') as f:

    cnt = 0
    for line in f:
        if cnt%1000 ==0:
            print(cnt)
        # if cnt > 2086000:
        if cnt >= 0:
            doc = nlp(line.strip().replace(","," "))
            # write_file.write(line)
            for token in doc:
                # print((token.head.text, token.text, token.pos_, token.dep_, spacy.explain(token.dep_)))
                # print(token.dep_, token.head.pos_,  token.pos_, token.head.text, token.text)
                # write_file.write(token.dep_ + "," + token.head.pos_ + "," + token.pos_ + "," + token.head.text + "," + token.text+"\n")
                write_file.write(token.pos_ + " ")
                # print(token.pos_)
                # write_file.write(token.dep_ + "," + token.head.pos_ + "," + token.pos_ +";")
            # write_file.write("----------------\n")
            write_file.write("\n")
            # write_file.write("\n----------------\n")
        cnt = cnt + 1

#public
#
# write_file1 = open("E:\\IdentifierQuality\\JDK17\\Patterns_parsedMethodNameTokens_public.txt", mode='w')
# with open("E:\\IdentifierQuality\\JDK17\\parsedMethodNameTokens_public.txt", 'r', encoding='utf-8') as f1:
# write_file1 = open("E:\\BadMethodName\\BadNames\\Dataset\\HighQualityProjects\\MethodNames\\trainingData_ngrams_POS.txt", mode='w')
# with open("E:\\BadMethodName\\BadNames\\Dataset\\HighQualityProjects\\MethodNames\\trainingData_ngrams.txt", 'r', encoding='utf-8') as f1:
#     cnt = 0
#     for line in f1:
#         if cnt % 1000 == 0:
#             print(cnt)
#         # if cnt > 2086000:
#         if cnt > 0:
#             doc = nlp(line.strip())
#             # write_file1.write(line)
#             for token in doc:
#                 # print((token.head.text, token.text, token.pos_, token.dep_, spacy.explain(token.dep_)))
#                 # print(token.dep_, token.head.pos_,  token.pos_, token.head.text, token.text)
#                 # write_file.write(token.dep_ + "," + token.head.pos_ + "," + token.pos_ + "," + token.head.text + "," + token.text+"\n")
#                 write_file1.write(token.pos_ + " ")
#                 # write_file.write(token.dep_ + "," + token.head.pos_ + "," + token.pos_ +";")
#             # write_file.write("\n----------------\n")
#             write_file1.write("\n")
#             # write_file1.write("----------------\n")
#         cnt = cnt + 1


'''
test each case
'''
# doc = nlp("get bean name")
# doc = nlp("get broker")
# doc = nlp("get total dests")
# doc = nlp("create temp log file")
# doc = nlp("write map")
# doc = nlp("get durable topic subscribers")
# displacy.serve(doc, style="dep")


# write_file1 = open("E:\BadMethodName\BadNames\Dataset\TSE21_42TestData\\Patterns_fixed_NarrowMeaning.txt", mode='w')
# with open("E:\BadMethodName\BadNames\Dataset\TSE21_42TestData\\methodNames_fixed_NarrowMeaning.txt", 'r', encoding='utf-8') as f1:



# write_file1 = open("E:\BadMethodName\BadNames\Dataset\TSE21_42TestData\\Patterns_buggy_NarrowMeaning.txt", mode='w')
# with open("E:\BadMethodName\BadNames\Dataset\TSE21_42TestData\\methodNames_buggy_NarrowMeaning.txt", 'r', encoding='utf-8') as f1:
#     cnt = 0
#     for line in f1:
#         print(line)
#         if cnt % 1000 == 0:
#             print(cnt)
#         # if cnt > 2086000:
#         if cnt >= 0:
#             doc = nlp(line.strip())
#             # write_file1.write(line)
#             for token in doc:
#                 # print((token.head.text, token.text, token.pos_, token.dep_, spacy.explain(token.dep_)))
#                 # print(token.dep_, token.head.pos_,  token.pos_, token.head.text, token.text)
#                 # write_file.write(token.dep_ + "," + token.head.pos_ + "," + token.pos_ + "," + token.head.text + "," + token.text+"\n")
#                 write_file1.write(token.pos_ + " ")
#                 # write_file1.write(token.dep_ + "," + token.head.pos_ + "," + token.pos_ +";")
#             # write_file.write("\n----------------\n")
#             write_file1.write("\n")
#             # write_file1.write("----------------\n")
#         cnt = cnt + 1



# print(spacy.explain("AUX"))
# print(spacy.explain("DET"))
# print(spacy.explain("PROPN"))
# print(spacy.explain("ADP"))
# print(spacy.explain("CCONJ"))
# print(spacy.explain("SCONJ"))
# print(spacy.explain("``"))
# print(spacy.explain("''"))