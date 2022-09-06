import bert_serving
import json
from bert_serving.client import BertClient
bc = BertClient()
vector_map = {}
token_set = set()
with open("E:\IdentifierQuality\BadNames\DataLabeling\\method_name_1124_split_balanced.txt",'r') as f:
    lines = f.readlines()
    token_set = lines[0].split(",")
    print(len(token_set))
    for token in token_set:
        print(token)
        vector = bc.encode([token])
        # vector = bc.encode(["get","package","mk","$"])
        print(vector)
        vector_map[token]=vector.tolist()

with open("./VectorMap_1124.json",'w') as f1:
    json_str = json.dumps(vector_map)
    f1.write(json_str)
    print(len(vector[0]))

