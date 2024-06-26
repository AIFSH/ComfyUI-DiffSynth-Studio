import os,sys
import time
import torch
from .util_nodes import now_dir,output_dir
sys.path.append(os.path.join(now_dir))

from diffsynth.extensions.RIFE import RIFESmoother
from diffsynth import ModelManager, SDVideoPipeline, ControlNetConfigUnit, VideoData, save_video, save_frames

import cuda_malloc
import folder_paths
from huggingface_hub import hf_hub_download



models_dir = os.path.join(now_dir, "models")
controlnet_dir = os.path.join(models_dir, "ControlNet")
animatediff_dir = os.path.join(models_dir,"AnimateDiff")
annotators_dir = os.path.join(models_dir, "Annotators")
sd_models_dir = os.path.join(models_dir, "stable_diffusion")
textual_inversion_dir = os.path.join(models_dir, "textual_inversion")
rife_dir = os.path.join(models_dir, "RIFE")

device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"

def get_4x_num(num):
    num_ = round(num)
    while num_ % 4 != 0:
        num_ -= 1
    return num_

class DiffTextNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True})
            }
        }
    RETURN_TYPES = ("TEXT",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "text"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_DiffSynth-Studio"

    def text(self,text):
        return (text,)

class SDPathLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "repo_id": ("STRING",{
                    "default": "philz1337x/flat2DAnimerge_v45Sharp"
                }),
                "filename":("STRING",{
                    "default": "flat2DAnimerge_v45Sharp.safetensors"
                })
            }}
    RETURN_TYPES = ("SD_MODEL_PATH",)
    FUNCTION = "load_checkpoint"

    CATEGORY = "AIFSH_DiffSynth-Studio"

    def load_checkpoint(self, ckpt_name,repo_id,filename):
        if ckpt_name:
            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        else:
        # download stable_diffusion from hf
            ckpt_path = hf_hub_download(repo_id=repo_id,
                            filename=filename,
                            local_dir=folder_paths.folder_names_and_paths["checkpoints"])
            
        return (ckpt_path,)

class ControlNetPathLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
            "model_id": ([
                "canny", "depth", "softedge", "lineart", "lineart_anime", "openpose", "tile"
            ]),
            "scale":("FLOAT",{
                "default": 0.5
            })
            }}

    RETURN_TYPES = ("ControlNetConfigUnit",)
    FUNCTION = "load_controlnet"

    CATEGORY = "AIFSH_DiffSynth-Studio"

    def load_controlnet(self, control_net_name,model_id,scale):
        filename = f"control_v11p_sd15_{model_id}.pth"
        if control_net_name:
            controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
            assert filename == control_net_name, f"{model_id} dismatch with {control_net_name}"
        else:
            controlnet_path = hf_hub_download(repo_id="lllyasviel/ControlNet-v1-1",
        
                                             filename=filename,local_dir=folder_paths.folder_names_and_paths["controlnet"])
        out_dict = {
            "model":ControlNetConfigUnit(
                    processor_id=model_id,
                    model_path=controlnet_path,
                    scale=scale
                ),
            "path":controlnet_path
        }
        return (out_dict,)

class VideoShadeNode:
    def __init__(self):
        try:
            # AnimateDiff
            hf_hub_download(repo_id="guoyww/animatediff",filename="mm_sd_v15_v2.ckpt",local_dir=animatediff_dir)
            # ControlNet
            hf_hub_download(repo_id="lllyasviel/ControlNet-v1-1",filename="control_v11f1e_sd15_tile.pth",local_dir=controlnet_dir)
            # Annotators
            hf_hub_download(repo_id="lllyasviel/Annotators",filename="sk_model.pth",local_dir=annotators_dir)
            hf_hub_download(repo_id="lllyasviel/Annotators",filename="sk_model2.pth",local_dir=annotators_dir)
            #textual_inversion
            hf_hub_download(repo_id="gemasai/verybadimagenegative_v1.3",filename="verybadimagenegative_v1.3.pt",local_dir=textual_inversion_dir)
            # RIFE
            hf_hub_download(repo_id="AlexWortega/RIFE",filename="flownet.pkl",local_dir=rife_dir)
        except:
            print("you can't attach huggingface? check your net and try again")
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "source_video_path": ("VIDEO",),
                "sd_model_path":("SD_MODEL_PATH",),
                "postive_prompt":("TEXT",),
                "negative_prompt":("TEXT",),
                "start":("INT",{
                    "default": 0
                }),
                "length":("INT",{
                    "default": -1
                }),
                "seed":("INT",{
                    "default": 42
                }),
                "cfg_scale":("INT",{
                    "default": 3
                }),
                "num_inference_steps":("INT",{
                    "default": 10
                }),
                "animatediff_batch_size":("INT",{
                    "default": 32
                }),
                "animatediff_stride":("INT",{
                    "default": 16
                }),
            },
            "optional":{
                "controlnet1":("ControlNetConfigUnit",),
                "controlnet2":("ControlNetConfigUnit",),
                "controlnet2":("ControlNetConfigUnit",),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "maketoon"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_DiffSynth-Studio"

    def maketoon(self,source_video_path,sd_model_path,postive_prompt,negative_prompt,start,length,seed,
                 cfg_scale,num_inference_steps,animatediff_batch_size,animatediff_stride,
                 controlnet1=None,controlnet2=None,controlnet3=None,):
        # load models
        model_manager = ModelManager(torch_dtype=torch.float16, device=device)
        model_manager.load_textual_inversions(textual_inversion_dir)
        controlnet_path_list = []
        controlnet_model_list = []
        if controlnet1:
            controlnet_path_list.append(controlnet1['path'])
            controlnet_model_list.append(controlnet1['model'])
        if controlnet2:
            controlnet_path_list.append(controlnet2['path'])
            controlnet_model_list.append(controlnet2['model'])
        if controlnet3:
            controlnet_path_list.append(controlnet3['path'])
            controlnet_model_list.append(controlnet3['model'])
        model_manager.load_models([
            sd_model_path,
            os.path.join(animatediff_dir,"mm_sd_v15_v2.ckpt"),
            os.path.join(rife_dir,"flownet.pkl")
        ]+controlnet_path_list)
        
        pipe = SDVideoPipeline.from_model_manager(
            model_manager,controlnet_config_units=controlnet_model_list
        )
        smoother = RIFESmoother.from_model_manager(model_manager)

        # Load video (we only use 60 frames for quick testing)
        # The original video is here: https://www.bilibili.com/video/BV19w411A7YJ/
        
        video = VideoData(video_file=source_video_path)
        org_h,org_w = video.shape
        height, width = (1024,get_4x_num(1024*org_w/org_h)) if org_h > org_w else (get_4x_num(1024*org_h/org_w),1024)
        print(f"orginal size: {org_h}X{org_w} \t resize: {height}X{width}")
        video.set_shape(height,width)

        fps = video.data.reader.get_meta_data['fps']
        duration = video.data.reader.get_meta_data['duration']
        assert start < duration and start + length < duration
        if length == -1:
            input_video = [video[i] for i in range(start*fps, (duration-start)*fps)]
        else:
            input_video = [video[i] for i in range(start*fps, (start+length)*fps)]

        # Toon shading (20G VRAM)
        torch.manual_seed(seed)
        output_video = pipe(
            prompt=postive_prompt,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale, clip_skip=2,
            controlnet_frames=input_video, num_frames=len(input_video),
            num_inference_steps=num_inference_steps, height=height, width=width,
            animatediff_batch_size=animatediff_batch_size, animatediff_stride=animatediff_stride,
            vram_limit_level=0,
        )
        output_video = smoother(output_video)

        # Save video
        outfile = os.path.join(output_dir,f'{time.time_ns()}_shaded' + os.path.basename(source_video_path))
        save_video(output_video, outfile, fps=fps)
        return (outfile,)