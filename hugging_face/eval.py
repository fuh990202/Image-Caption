from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model= r'RobertaMLM',
    tokenizer= 'Byte_tokenizer'
)

print(fill_mask("a girl going into a <mask> building"))