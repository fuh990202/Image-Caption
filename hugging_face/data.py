import utils

# Preprocess training data
image_caption_dict = utils.prepare_dataset()
train, test = utils.train_test_split(image_caption_dict)
train_df, test_df = utils.generate_df(train), utils.generate_df(test)