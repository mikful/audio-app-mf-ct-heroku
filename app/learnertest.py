from fastai2 import *
from fastai2.vision.all import *
from fastai2_audio.core.all import *
from fastai2_audio.augment.all import *
from pydub import AudioSegment

# import all additional Learner functions
from utils import *

import json


testfile = "/home/mikef/Documents/ml-test-files/audio/Cat-Meow.wav"
model = "/home/mikef/Documents/GitHub/audio-app-mf-ct/app/models/xresnet18-mixup-80epoch-moredata_augsv3.pkl"

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