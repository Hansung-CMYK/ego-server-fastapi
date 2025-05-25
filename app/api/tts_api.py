import importlib, os, sys
from fastapi import APIRouter, HTTPException, UploadFile, File
import shutil

here = os.path.dirname(__file__)
api_file_path = os.path.abspath(os.path.join(here, "../../modules/GPT-SoVITS/api_v2.py"))
gpt_sovits_root = os.path.dirname(api_file_path)
gpt_sovits_sub  = os.path.join(gpt_sovits_root, "GPT_SoVITS")

for path in (gpt_sovits_root, gpt_sovits_sub):
    if path not in sys.path:
        sys.path.insert(0, path)

spec = importlib.util.spec_from_file_location("gpt_sovits_api", api_file_path)
gpt_sovits_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpt_sovits_api)

sys.modules["gpt_sovits_api"] = gpt_sovits_api

gsv = importlib.import_module("gpt_sovits_api")

router = APIRouter()

for route in gpt_sovits_api.APP.routes:
    if route.path == "/":
        continue
    router.routes.append(route)

async def update_refer(model_id: str, refer_path: str, refer_text: str, refer_language: str) -> None:
    if model_id not in gsv.speaker_list:
        raise ValueError(f"Model {model_id} not found.")
    
    speaker = gsv.speaker_list[model_id]
    
    speaker.default_refer = gsv.DefaultRefer(refer_path, refer_text, refer_language)
    print(f"Refer updated for model {model_id}")

@router.post("/update_refer/{model_id}")
async def api_update_refer(model_id: str = "default", refer_path: str = "/home/keem/refer/", refer_text: str = "/home/keem/refer/", refer_language: str = "ko"):
    try:
        await update_refer(model_id, refer_path, refer_text, refer_language)
        return {"message": f"Refer updated for model {model_id}"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
@router.post("/upload_wav/")
async def upload_wav(file: UploadFile = File(...), filename: str = None):
    if '.wav' not in filename:
        filename += ".wav"
    save_dir = "/home/keem/refer/"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if not filename:
        filename = file.filename

    file_path = os.path.join(save_dir, filename)
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"message": f"File {filename} uploaded and saved to {file_path}"}