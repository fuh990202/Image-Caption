from transformers import VisionEncoderDecoderModel
from PIL import Image
from transformers import RobertaTokenizerFast
from transformers import ViTFeatureExtractor

import data

MAX_LEN = 128 

trainer = VisionEncoderDecoderModel.from_pretrained('Image_Cationing_VIT_Roberta_iter2')
temp = data.test_df.sample(1).images.iloc[0]
Image.open(temp).convert("RGB")

tokenizer = RobertaTokenizerFast.from_pretrained('Byte_tokenizer', max_len=MAX_LEN)
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
caption = tokenizer.decode(trainer.generate(feature_extractor(Image.open(temp).convert("RGB"), return_tensors="pt").pixel_values)[0])
print(caption)