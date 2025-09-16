from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = '899b3ed3-3436-431d-8a6f-d8f1ce11b08e'

api = HubApi()
api.login(YOUR_ACCESS_TOKEN)
api.push_model(
    model_id="yszbb61255873/stable-diffusion-v1.5-inpainting",
    model_dir="F:\\models\\stable-diffusion-v1.5-inpainting"  # 本地模型目录，要求目录中必须包含configuration.json
)
