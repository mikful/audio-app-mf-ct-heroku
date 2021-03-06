import os
import requests
import aiohttp
import asyncio
import uvicorn
import ast
import aiofiles
from torch.distributions.beta import Beta
from fastai import *
from fastai.vision.all import *
from fastaudio.core.all import *
from fastaudio.augment.all import *
from fastai.callback.all import *
from io import BytesIO
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json

# import all additional Learner functions
from utils import *

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'],allow_headers=["*"])
app.mount('/static', StaticFiles(directory='app/static'), name="static")

path = Path(__file__).parent
export_file_url = 'https://storage.googleapis.com/fastai-export-bucket/v1-xresnet18-80epoch-standard-cutmix%2Bmixup.pkl' # google cloud bucket
export_file_name = 'export.pkl'


async def download_file(url, dest):
    if dest.exists(): 
        return "dest.exists()"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                f = await aiofiles.open(dest, mode='wb')
                await f.write(await response.read())
                await f.close()

async def setup_learner():
    pkl_dest = path/"models"/export_file_name
    await download_file(export_file_url, pkl_dest)
    try:
        learn = load_learner(pkl_dest)
        learn.dls.device = 'cpu'
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


learn = None
@app.on_event("startup")
async def startup_event():
    """Setup the learner on server start"""
    global learn
    loop = asyncio.get_event_loop()  # get event loop
    tasks = [asyncio.ensure_future(setup_learner())]  # assign some task
    learn = (await asyncio.gather(*tasks))[0]  # get tasks


@app.get('/')
async def homepage():
    html_content = (path / 'view' / 'index.html').open().read()
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/analyze")
async def analyze(file: bytes = File(...)):
    wav = BytesIO(file)
    utc_time = str(int(time.time()))
    sound_file = "tmp/sound_" + utc_time + ".wav"
    with open(sound_file, mode='bx') as f: f.write(wav.getvalue())
    prediction, idx, preds =  learn.predict(Path(sound_file))
    predictions_ordered = learn.dls.vocab[np.argsort(preds.squeeze()).squeeze()][::-1] # descending order
    conf_sorted = np.sort(preds.squeeze()).squeeze()[::-1] # descending order
    results_ordered = tuple(zip(predictions_ordered, np.rint(conf_sorted*100).tolist()))
    return JSONResponse({'classifications': json.dumps(results_ordered)})


Port = int(os.environ.get('PORT', 5000))
if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=Port, log_level="info") #heroku
