import json
def getJsonData (json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        contents = f.read()
        while "/*" in contents:
            preComment, postComment = contents.split("/*", 1)
            contents = preComment + postComment.split("*/", 1)[1]
        json_data = json.loads(contents.replace("'", '"'))
    return json_data