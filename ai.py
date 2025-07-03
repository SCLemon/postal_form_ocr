from transformers import AutoModel, AutoTokenizer, GenerationConfig
import torch
from mtkresearch.llm.prompt import MRPromptV3

# 全域變數，載入一次
model = None
tokenizer = None
prompt_engine = MRPromptV3()

def load_model(model_path):
    global model, tokenizer
    if model is None or tokenizer is None:
        print(f"載入模型：{model_path}")
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='cuda',
            low_cpu_mem_usage=True,
            img_context_token_id=128212
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        print("模型載入完成")

def run(url, model_path='./local_breeze2_3b'):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    load_model(model_path)  # 只會第一次載入，之後不會重複

    generation_config = GenerationConfig(
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.01,
        top_p=0.01,
        repetition_penalty=1.1,
        eos_token_id=128009
    )

    sys_prompt = 'You are a helpful AI assistant built by MediaTek Research. The user you are helping speaks Traditional Chinese and comes from Taiwan.'

    def _inference(prompt, pixel_values=None):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        if pixel_values is None:
            output = model.generate(**inputs, generation_config=generation_config)
        else:
            output = model.generate(
                **inputs,
                generation_config=generation_config,
                pixel_values=pixel_values.to(model.device, dtype=model.dtype)
            )
        return tokenizer.decode(output[0], skip_special_tokens=True)

    conversations = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            {"type": "image", "image_path": url},
            {"type": "text", "text": "請判斷整張圖片的文字和數字，切記不要輸出換行符號, 其餘無需回覆。"},
        ]},
    ]

    prompt, pixel_values = prompt_engine.get_prompt(conversations)
    output_str = _inference(prompt, pixel_values=pixel_values)

    result = prompt_engine.parse_generated_str(output_str)
    return result["content"]
