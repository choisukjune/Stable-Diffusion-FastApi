#!/usr/bin/python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import fastapi as _fapi

import schemas as _schemas
import services as _services
import io
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to Stable Diffussers API"}

# Endpoint to test the Front-end and backend
@app.get("/api")
async def root():
    return {"message": "Welcome to the Demo of StableDiffusers with FastAPI"}

@app.get("/api/generate/")
async def generate_image(imgPromptCreate: _schemas.ImageCreate = _fapi.Depends()):
    
    image = await _services.generate_image(imgPrompt=imgPromptCreate)

    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")


#uvicorn main:app --reload --host=0.0.0.0 --port=80
# if __name__ == "__main__":
#     uvicorn.run('main:app', host="0.0.0.0", port=80, reload=True )
#     #uvicorn.run('main:app', host="0.0.0.0", port=80, reload=True, access_log=False, log_config="log.yaml", reload_excludes="api.log" )
#     #uvicorn.run('main:app', host="0.0.0.0", port=80, reload=True, access_log=True )
