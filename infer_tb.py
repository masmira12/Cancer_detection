from fastai import *
from fastai.vision import *
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--path", type=str, required=True)
ap.add_argument("--model", type=str, required=True)
ap.add_argument("--image", type=str, required=True)
args=ap.parse_args()

#learn = load_learner(args.path, fname="../models/{}".format(args.model))
learn = load_learner(args.path).to_fp16()
#learn = load_learner().to_fp16()
learn.model.cuda()
preds = learn.predict(open_image(args.image))
#print(preds)

PREDS_CLASS = str(preds[0])
PREDS_PROB = str(round(np.max(preds[2].numpy()),2)*100)[0:5]

output = {"predicted_class":PREDS_CLASS, "predicted_prob":PREDS_PROB}
print(output, end='')
