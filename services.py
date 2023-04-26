from pathlib import Path 
import schemas as _schemas

import torch 
from diffusers import StableDiffusionPipeline
from PIL.Image import Image
import os

# load_dotenv()

# # Get the token from HuggingFace 
# """
# Note: make sure .env exist and contains your token
# """
# HF_TOKEN = os.getenv('HF_TOKEN')

# # Create the pipe 
# pipe = StableDiffusionPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4", 
#     revision="fp16", 
#     torch_dtype=torch.float16,
#     use_auth_token=HF_TOKEN
#     )

# model_id = "akurei/waifu-diffusion"
"""
~/.cache/huggingface/diffusers/ 
모델을 삭제한다.
https://sigmasabjil.tistory.com/22
1. 폴더 찾기
1) 전체 폴더에서 찾기 
find / -name 폴더명 -type d 

2)현재폴더(하위 포함) 에서 찾기
find ./ -name 폴더명 -type d

2. 파일 찾기(ls)
 

1) ls로 찾기


ls -Rhal | grep '.*[.]html'  <---- 확장자가 html인것 찾음

 

 

R:하위폴더 포함

h: 사람이 보기쉽게 해줌

a: 숨겨진 파일,디렉토리 보여줌

l: 자세히 보여줌(퍼미션,소유자,그룹..)



-찾은 파일 삭제(주의!!!)

 
ls | grep '키워드' | xargs rm   <---키워드 찾아서 삭제


2) find 로 찾기


find 경로(찾을범위) 경로옵션 ,옵션조건( ex 찾을값) 

경로옵션: -name, -user(소유자), -newer -perm -size  -type
옵션조건:  -print(기본값) -exec(외부 명령실행) 

find /etc -name "*.conf"   <--  /etc 디렉토리 하위에 확장명이 .conf인 파일검색 

 

sudo find / -name "xxx" -type f     xxx 이름의 파일 찾기

에러나는거 날리고 찾기

ex) sudo  find / -name "xxx" -print 2>/dev/null
ex) sudo find / -name "com.trendmicro*" -print 2>/dev/null

 

 

-찾은 파일 삭제(주의!!!)

 

ex)

find /home -name "*.swp" -exec rm {} \;   /home 홈 디렉토리 하위에 확장명이 *.swp인 파일을 삭제 
-exec : 외부명령의 시작  ,  \; 외부 명령의 끝   , {}는 앞에서 find 명령의 실행결과물들어가는 곳 


find . -type f -name "*2017*" -exec rm {} \;   현재디렉토리에서 2017 들어간거 전부 삭제


cat /dev/null > catalina.out 카탈리나  빈파일로 만들기 


3. 내부 내용 찾기 
 

1)파일 내부에 문자열을 검색함



grep 'meta' ./*.html  

내부에 meta라는 문자열이 들어있는 html파일 찾음


grep  -rn 'meta' ./* 

하위 모든 디렉토리 파일대상으로 해당 문자열 찾음

-r은 하위 도 포함 
-n 라인넘버 표시 


tail -1000 /usr/local/tomcat/logs/catalina.out | grep 'error' 



2) 문자와 문자 사이의 내용 출력


sed -n -e '/Word A/,/Word D/ p' file 

Word A
Word B
Word C
Word D
Word E
Word F

It seems that you want to print lines between 'Word A' and 'Word D' (inclusive). I suggest you to use sed instead of grep. It lets you to edit a range of input stream which starts and ends with patterns you want. You should just tell sed to print all lines in range and no other lines:

sed -n -e '/2019-06-14 04:07/,/2019-06-14 04:11/ p' catalina.out 


3) 내용 역순으로 출력


tail -500 /usr/local/tomcat/logs/catalina.out | tac

카탈리나 거꾸로 보기 밑을 위로



4) 해당 문자열 찾은 후  위아래 몇줄 추가로 보여줌

tail -1000 /usr/local/tomcat/logs/catalina.out | grep -A10 -B10 'error'

grep -A10 -B10 '찾을 단어'  

-B 숫자,  print NUM lines of leading context (찾은 라인의 위줄을 숫자만큼 추가로 보여줌)
-A 숫자,  print NUM lines of trailing context (찾은 라인의  아래줄을 숫자만큼 추가로 보여줌)
"""
model_id = "prompthero/openjourney"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
# pipe = StableDiffusionPipeline.from_pretrained(
#     'prompthero/openjourney',
#     revision="fp16", 
#     torch_dtype=torch.float16,
# )
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# if torch.backends.mps.is_available():
#     device = "mps"
# else: 
#     device = "cuda" if torch.cuda.is_available() else "cpu"

device = "cuda"

pipe.to(device)


async def generate_image(imgPrompt: _schemas.ImageCreate) -> Image: 
    #generator = None if imgPrompt.seed is None else torch.Generator().manual_seed(int(imgPrompt.seed))
    generator = torch.Generator(device=device)
    image: Image = pipe(imgPrompt.prompt,
                        guidance_scale=imgPrompt.guidance_scale, 
                        num_inference_steps=imgPrompt.num_inference_steps, 
                        generator = generator,
                        # height=512, width=768
                    ).images[0]
    
    return image