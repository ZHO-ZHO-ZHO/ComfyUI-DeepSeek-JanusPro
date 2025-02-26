import sys
import os
import torch
import numpy as np
import folder_paths
import time
import re
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM

# 关键路径处理：将当前目录添加到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # 添加当前目录到Python路径

try:
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    from janus.utils.io import load_pil_images
except ImportError as e:
    print(f"路径调试信息：")
    print(f"当前目录: {current_dir}")
    print(f"目录内容: {os.listdir(current_dir)}")
    print(f"sys.path: {sys.path}")
    raise

# 添加模型路径配置
current_directory = os.path.dirname(os.path.abspath(__file__))
folder_paths.folder_names_and_paths["Janus"] = ([os.path.join(folder_paths.models_dir, "Janus")], folder_paths.supported_pt_extensions)

GPU_ARCHITECTURE_CUDA = "cuda"
GPU_ARCHITECTURE_MPS = "mps"
GPU_ARCHITECTURE_CPU = "cpu"
gpu_architecture = None

if torch.cuda.is_available():
    gpu_architecture = GPU_ARCHITECTURE_CUDA
elif torch.mps.is_available() :
    gpu_architecture = GPU_ARCHITECTURE_MPS
else :
    gpu_architecture = GPU_ARCHITECTURE_CPU

device = torch.device(gpu_architecture)

# 辅助函数
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class Janus_ModelLoader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "deepseek-ai/Janus-Pro-7B"}),
            }
        }

    RETURN_TYPES = ("JANUS_MODEL", "PROCESSOR", "TOKENIZER")
    RETURN_NAMES = ("model", "processor", "tokenizer")
    FUNCTION = "load_model"
    CATEGORY = "🧩Janus"

    def load_model(self, model_path):
        # 加载配置
        config = AutoConfig.from_pretrained(model_path)
        language_config = config.language_config
        language_config._attn_implementation = 'eager'

        # 加载模型
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_path,
            language_config=language_config,
            trust_remote_code=True
        ).to(torch.bfloat16 if gpu_architecture == GPU_ARCHITECTURE_CUDA or gpu_architecture == GPU_ARCHITECTURE_MPS else torch.float16).to(device).eval()

        # 加载处理器
        processor = VLChatProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer

        return (vl_gpt, processor, tokenizer)

class Janus_MultimodalUnderstanding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("JANUS_MODEL",),
                "processor": ("PROCESSOR",),
                "tokenizer": ("TOKENIZER",),
                "image": ("IMAGE",),
                "question": ("STRING", {"default": "describe the image", "multiline": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "max_new_tokens": ("INT", {"default": 512, "min": 16, "max": 2048}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "understand"
    CATEGORY = "🧩Janus"

    def understand(self, model, processor, tokenizer, image, question, seed, top_p, temperature, max_new_tokens=512):
        # 修复种子范围问题
        seed = seed % (2**32)
        
        # 设置随机种子（添加CUDA同步）Works with MPS
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))  # 适配numpy种子范围
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.synchronize()

        try:
            # 图像预处理（添加维度验证）
            if isinstance(image, list):
                image_tensor = image[0]
            else:
                image_tensor = image
                
            pil_image = tensor2pil(image_tensor)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # 构建对话（添加异常处理）
            try:
                conversation = [{
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{question}",
                    "images": [pil_image],
                }, {
                    "role": "<|Assistant|>", 
                    "content": ""
                }]
            except Exception as e:
                print(f"对话构建失败: {e}")
                return ("Error: Invalid conversation format",)

            # 处理输入（添加维度调试）
            try:
                prepare_inputs = processor(
                    conversations=conversation,
                    images=[pil_image],
                    force_batchify=True
                ).to(model.device, dtype=torch.bfloat16 if gpu_architecture == GPU_ARCHITECTURE_CUDA or gpu_architecture == GPU_ARCHITECTURE_MPS else torch.float16)
                
                print(f"输入张量形状 - input_ids: {prepare_inputs.input_ids.shape}")
                print(f"注意力掩码形状: {prepare_inputs.attention_mask.shape}")
            except Exception as e:
                print(f"输入处理失败: {e}")
                return ("Error: Input processing failed",)

            # 生成过程（添加参数验证）
            try:
                inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
                print(f"输入嵌入形状: {inputs_embeds.shape}")

                generation_config = {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": prepare_inputs.attention_mask,
                    "pad_token_id": tokenizer.eos_token_id,
                    "bos_token_id": tokenizer.bos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": temperature > 0,
                    "temperature": temperature if temperature > 0 else 1.0,
                    "top_p": top_p,
                }

                # 执行生成（添加时间监控）
                start_time = time.time()
                outputs = model.language_model.generate(**generation_config)
                print(f"生成耗时: {time.time() - start_time:.2f}秒")

            except Exception as e:
                print(f"生成失败: {e}")
                return ("Error: Generation failed",)

            # 解码输出（添加异常处理）
            try:
                full_output = outputs[0].cpu().tolist()
                answer = tokenizer.decode(full_output, skip_special_tokens=True)
                
                # 清理特殊标记
                clean_pattern = r'<\|.*?\|>'
                clean_answer = re.sub(clean_pattern, '', answer).strip()
                
                return (clean_answer,)
                
            except Exception as e:
                print(f"解码失败: {e}")
                return ("Error: Output decoding failed",)

        except Exception as e:
            print(f"处理过程中出现未捕获的异常: {e}")
            return ("Error: Unexpected processing error",)


class Janus_ImageGeneration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("JANUS_MODEL",),
                "processor": ("PROCESSOR",),
                "tokenizer": ("TOKENIZER",),
                "prompt": ("STRING", {"multiline": True, "default": "Master shifu racoon wearing drip attire"}),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 0xffffffffffffffff}),
                "cfg_weight": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0, "step": 0.5}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "🧩Janus"

    def generate(self, model, processor, tokenizer, prompt, seed, cfg_weight, temperature):
        # 清理缓存并设置种子
        torch.cuda.empty_cache()
        torch.mps.empty_cache()
        seed = seed % (2**32)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 固定参数（与原始代码一致）
        width = 384
        height = 384
        parallel_size = 5
        patch_size = 16
        image_token_num = 576

        # 构建输入文本
        messages = [{'role': '<|User|>', 'content': prompt},
                   {'role': '<|Assistant|>', 'content': ''}]
        text = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=messages,
            sft_format=processor.sft_format,
            system_prompt=''
        ) + processor.image_start_tag

        # 生成输入ID
        input_ids = torch.LongTensor(tokenizer.encode(text)).to(model.device)

        # 初始化Tokens（严格保持原始结构）
        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int, device=model.device)
        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = processor.pad_id

        # 生成过程（保持原始循环结构）
        inputs_embeds = model.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros((parallel_size, image_token_num), dtype=torch.int, device=model.device)
        
        pkv = None
        for i in range(image_token_num):
            with torch.no_grad():
                outputs = model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=pkv
                )
                pkv = outputs.past_key_values
                
                # 原始分类器自由引导实现
                logits = model.gen_head(outputs.last_hidden_state[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                
                # 采样逻辑
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, i] = next_token.squeeze(dim=-1)
                
                # 准备下一轮输入（保持原始视图操作）
                next_token = torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1).view(-1)
                img_embeds = model.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1)

        # 图像解码（严格保持原始实现）
        patches = model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, width//patch_size, height//patch_size]
        )
        
        # 后处理（原始unpack逻辑）
        dec = patches.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
        visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        # 转换为ComfyUI图像格式
        output_images = []
        for i in range(parallel_size):
            pil_img = Image.fromarray(visual_img[i]).resize((768, 768), Image.LANCZOS)
            output_images.append(pil2tensor(pil_img))
        
        return (torch.cat(output_images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "Janus_ModelLoader": Janus_ModelLoader,
    "Janus_MultimodalUnderstanding": Janus_MultimodalUnderstanding,
    "Janus_ImageGeneration": Janus_ImageGeneration
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Janus_ModelLoader": "🧩Janus Model Loader",
    "Janus_MultimodalUnderstanding": "🧩Janus Multimodal Understanding",
    "Janus_ImageGeneration": "🧩Janus Image Generation"
}
