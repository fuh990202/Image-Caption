# Decoder
from transformers import RobertaTokenizerFast # After training tokenizern we will wrap it so it can be used by Roberta model



#Encoder-Decoder Model
from transformers import VisionEncoderDecoderModel

#Training
# When using previous version of the library you need the following two lines
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments


from transformers import RobertaTokenizerFast
from transformers import ViTFeatureExtractor
from transformers import default_data_collator

import data
import dataset
import utils
import datasets



TRAIN_BATCH_SIZE = 20   # input batch size for training (default: 64)
VALID_BATCH_SIZE = 5   # input batch size for testing (default: 1000)
LEARNING_RATE = 1e-4    # learning rate (default: 0.01)
MAX_LEN = 128           # Max length for product description
TRAIN_EPOCHS = 2       # number of epochs to train (default: 10)
WEIGHT_DECAY = 0.01


def train():
    # load decoder tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('Byte_tokenizer', max_len=MAX_LEN)

    # load pretrained vision transformer Feature Extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    # create datasets
    train_dataset = dataset.IAMDataset(df=data.train_df.sample(frac=0.3,random_state=2).iloc[:10000].reset_index().drop('index',axis =1),
                            tokenizer=tokenizer,
                            feature_extractor= feature_extractor)
    eval_dataset = dataset.IAMDataset(df=data.test_df.sample(frac=0.1,random_state=2)[:2000].reset_index().drop('index',axis =1),
                            tokenizer=tokenizer,feature_extractor= feature_extractor)


    # Conect Encoder and Decoder model using VisionEncoderDecoder
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained\
                        ("google/vit-base-patch16-224-in21k", 'RobertaMLM', tie_encoder_decoder=True)


    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.max_length = 20
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    def compute_metrics(pred):
        rouge = datasets.load_metric("rouge")
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # all unnecessary tokens are removed
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }
    # set training arguments
    captioning_model = 'VIT_Captioning'
    training_args = Seq2SeqTrainingArguments(
        output_dir=captioning_model,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=VALID_BATCH_SIZE,
        predict_with_generate=True,
        #evaluate_during_training=True,
        evaluation_strategy="epoch",
        do_train=True,
        do_eval=True,
        logging_steps=1024,  
        save_steps=2048, 
        warmup_steps=1024,  
        #max_steps=1500, # delete for full training
        num_train_epochs = TRAIN_EPOCHS, #TRAIN_EPOCHS
        overwrite_output_dir=True,
        save_total_limit=1,
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        tokenizer=feature_extractor,
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    # Fine-tune the model, training and evaluating on the train dataset
    trainer.train()

    trainer.save_model('Image_Cationing_VIT_Roberta_iter2')

if __name__ == '__main__':
    train()