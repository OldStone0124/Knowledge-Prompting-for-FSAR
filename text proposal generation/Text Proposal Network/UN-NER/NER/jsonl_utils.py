import json
from ipdb import set_trace

def load_jsonl(pth, check_label_empty=False):
    ret = []
    with open(pth, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        if check_label_empty:
            if len(result['label'])==00000000:
                continue
        # print(f"result: {result}")
        # print(isinstance(result, dict))
        ret.append(result)
    return ret

if __name__=='__main__':
    loaded = load_jsonl("annotations/4_20220427.jsonl")
    set_trace()