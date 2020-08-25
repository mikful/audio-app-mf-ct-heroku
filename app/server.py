import os
import aiohttp
import asyncio
import uvicorn
import ast
import aiofiles
from fastai2 import *
from fastai2.vision.all import *
from fastai2_audio.core.all import *
from fastai2_audio.augment.all import *
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
export_file_url = 'https://storage.googleapis.com/fastai-export-bucket/xresnet18-mixup-80epoch-moredata_augsv3.pkl' # google cloud bucket
export_file_name = 'export.pkl'

# with open('app/classes.txt', 'r') as f:
#     classes = ast.literal_eval(f.read())


async def download_file(url, dest):
    print("Attempting pkl file download")
    print("url:", url)
    print("dest:", dest)
    if dest.exists(): 
        return "dest.exists()"
    async with aiohttp.ClientSession() as session:
        print("async session")
        async with session.get(url) as response:
            print("response", response)
            if response.status == 200:
                f = await aiofiles.open(dest, mode='wb')
                print("writing learner data:", f)
                await f.write(await response.read())
                await f.close()

async def setup_learner():
    pkl_dest = path/"models"/export_file_name
    await download_file(export_file_url, pkl_dest)
    try:
        print("pkl file exists?:", path/export_file_name, os.path.exists(pkl_dest))
        print("dl pkl file size:", Path(pkl_dest).stat().st_size)
        print("loading learner...")
        learn = load_learner(pkl_dest)
        learn.dls.device = 'cpu'
        print("learner loaded")
        print("learner classes:", learn.dls.vocab)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
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
    print("wav bytes content:", wav)
    print(type(wav))
    utc_time = str(int(time.time()))
    sound_file = "audio_uploads/sound_" + utc_time + ".wav"
    with open(sound_file, mode='bx') as f: f.write(wav.getvalue())
    print("sound_file:", sound_file)
    print("audio file size:", Path(sound_file).stat().st_size)
    test_dl = learn.dls.test_dl(sound_file, with_label=False) # use tta for higher accuracy
    with learn.no_bar():
        preds, targs = learn.tta(dl=test_dl)
    print("preds:", preds)
    #predictions = learn.dls.vocab[np.argwhere(preds.squeeze() > 0.2).squeeze()] # 20% threshold (maybe use later)
    predictions_ordered = learn.dls.vocab[np.argsort(preds.squeeze()).squeeze()][::-1] # descending order
    conf_sorted = np.sort(preds.squeeze()).squeeze()[::-1] # descending order
    results_ordered = tuple(zip(predictions_ordered, np.rint(conf_sorted*100).tolist()))
    print(f"first 5 predictions_ordered: {results_ordered[:5]}", )
    return JSONResponse({'classifications': json.dumps(results_ordered)})


# if __name__ == '__main__':
#     if 'serve' in sys.argv:
#         uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info") # render
#         #uvicorn.run(app=app, host='0.0.0.0', port=Port, log_level="info") #heroku
if __name__ == '__main__':
    port = os.getenv('PORT',5000)
    app.run(debug=True, host='0.0.0.0', port=port) 
