
import json
import random
import pandas as pd

def prepare_dataset(json_file='data.json', images_path = '../data/images/'):
    # descriptive caption dataset
    # with open('/content/gdrive/MyDrive/Colab Notebooks/project/data.json', 'r') as openfile:
    #     json_object = json.load(openfile)
    # images_caption_dict = dict(json_object)
    # images_path = "/content/gdrive/MyDrive/Colab Notebooks/project/images/"


    # manual labelled dataset
    # with open('/content/gdrive/MyDrive/Colab Notebooks/project/data1.json', 'r') as openfile:
    #     json_object1 = json.load(openfile)
    # images_caption_dict.update(dict(json_object1))
    # images_path = "/content/gdrive/MyDrive/Colab Notebooks/project/images/"

    # instagram dataset
    with open('/content/gdrive/MyDrive/Colab Notebooks/project/ins_data/ins_data.json', 'r') as openfile:
        json_object = json.load(openfile)
    images_caption_dict = dict(json_object)
    images_path = "/content/gdrive/MyDrive/Colab Notebooks/project/ins_data/img/"

    images = list(images_caption_dict.keys())

    for image_path in images:
        if image_path.endswith('jpg'):
            new = images_path + image_path.split('/')[-1]
            images_caption_dict[new] = images_caption_dict.pop(image_path)
        else:
            images_caption_dict.pop(image_path)

    return images_caption_dict


def train_test_split(dictionary):
    images = dictionary.keys()
    images_test = random.sample(images,int(0.3*len(images)))
    images_train = [img for img in images if img not in images_test]

    train_dict = {
      img: dictionary[img] for img in images_train
    }

    test_dict = {
      img: dictionary[img] for img in images_test
    }
    return(train_dict,test_dict)

def generate_df(dictionary):
    df = pd.DataFrame([])

    captions = []
    images = []
    for image in list(dictionary.keys()):
        caption = dictionary[image]
    #     captions.append(('.'.join([ sent.rstrip() for sent in ('.'.join(caption)).split('<e>.<s>')]))\
    #                             .replace('<s> ','').replace('  <e>','.'))
        for capt in caption:
            captions.append(' '.join(capt.replace('<s> ','').replace('  <e>','').strip().split(' ')[:30]))
            images.append(image)

    df['images'] = images
    df['captions'] = captions
    return(df)