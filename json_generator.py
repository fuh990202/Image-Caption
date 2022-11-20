import json

def generate_baseline_datasets(origin_json_file, target_json_file, n_train, n_val, n_test):
    with open (origin_json_file, 'r') as j:
        data = json.load(j)

    trainImages, valImages, testImages = [], [], []
    for img in data['images']:
        if img['split'] in {'train', 'restval'}:
            trainImages.append(img)
        elif img['split'] in {'val'}:
            valImages.append(img)
        elif img['split'] in {'test'}:
            testImages.append(img)

    partial_json = data
    partial_json['images'] = trainImages[:n_train]+valImages[:n_val]+testImages[:n_test]
    with open(target_json_file, 'w') as j:
        json.dump(partial_json, j)

def generate_anticipated_datasets(baseline_json_file, target_json_file, path):
    with open (baseline_json_file, 'r') as j:
        data = json.load(j)
    
    target_json = {}
    for img in data['images']:
        key = path + img['filepath'] + '/' + img['filename']
        value = []
        for sentence in img['sentences']:
            value.append('<s> ' + sentence['raw'] + ' <e>')
        target_json[key] = value

    with open(target_json_file, 'w') as j:
        json.dump(target_json, j)


origin_json_file = './data/dataset_coco.json'
baseline_json_file = './data/baseline_dataset.json'
anticipated_json_file = './data/anticipated_dataset.json'

generate_baseline_datasets(origin_json_file, baseline_json_file, 800, 100, 100)
# generate_anticipated_datasets(baseline_json_file, anticipated_json_file, '../data/')
