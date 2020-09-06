from fastai import *
from fastai.vision.all import *
from fastaudio.core.all import *
from fastaudio.augment.all import *

# import all additional Learner functions
from utils import *
from pathlib import Path

import json


testfile = Path("/workspaces/audio-app-mf-ct-heroku/app/testfiles/Cat-Meow.wav")
model = "/workspaces/audio-app-mf-ct-heroku/app/models/v1-xresnet18-80epoch-standard-cutmix+mixup.pkl"

def analyze():
    learn = load_learner(model)
    prediction, idx, preds =  learn.predict(testfile)
    print("prediction:", prediction)
    print("preds:", preds)
    #predictions = learn.dls.vocab[np.argwhere(preds > 0.1).squeeze()] # 10% threshold
    predictions_ordered = learn.dls.vocab[np.argsort(preds).squeeze()][::-1] # descending order
    conf_sorted = np.sort(preds).squeeze()[::-1] # descending order
    results_ordered = tuple(zip(predictions_ordered, np.rint(conf_sorted*100).tolist()))
    #print(f"first 5 predictions_ordered: {list(results_ordered)[:5]}", )
    #print(json.dumps(str(results_ordered)))
    return json.dumps(results_ordered)


def analyzetta():
    learn = load_learner(model)
    test_dl = learn.dls.test_dl(testfile, with_label=False)
    with learn.no_bar():
        preds, targs = learn.tta(dl=test_dl)
    print("preds:", preds)
    print("targs:", targs)
    #predictions = learn.dls.vocab[np.argwhere(preds > 0.1).squeeze()] # 10% threshold
    predictions_ordered = learn.dls.vocab[np.argsort(preds.squeeze()).squeeze()][::-1] # descending order
    conf_sorted = np.sort(preds).squeeze()[::-1] # descending order
    results_ordered = tuple(zip(predictions_ordered, np.rint(conf_sorted*100).tolist()))
    #print(f"first 5 predictions_ordered: {list(results_ordered)[:5]}", )
    #print(json.dumps(str(results_ordered)))
    return json.dumps(results_ordered)

print("predict method")
print(analyze())
print()
print("tta method")
print(analyzetta())