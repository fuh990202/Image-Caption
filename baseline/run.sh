'''
requirement to run this script:

.local/lib/python3.10/site-packages/torch/optim/optimizer.py
comment out the line below
# self.defaults.setdefault('differentiable', False)


'''

MODEL_PATH='./checkpoint_wordmap/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
WORD_MAP_PATH='./checkpoint_wordmap/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'

python3 ./app/caption.py --img='./image/image1.jpg' --model=$MODEL_PATH --word_map=$WORD_MAP_PATH --beam_size=5