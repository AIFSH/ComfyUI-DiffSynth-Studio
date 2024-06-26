import os,sys
import shutil
import time,math
import torch
from .util_nodes import now_dir,output_dir
sys.path.append(os.path.join(now_dir))

from diffsynth.extensions.RIFE import RIFESmoother
from diffsynth import ModelManager, SDVideoPipeline, ControlNetConfigUnit, VideoData, save_video, save_frames

import cuda_malloc
import folder_paths
from huggingface_hub import hf_hub_download

models_dir = os.path.join(now_dir, "models")
animatediff_dir = os.path.join(models_dir,"AnimateDiff")
annotators_dir = os.path.join(folder_paths.models_dir, "Annotators")
textual_inversion_dir = os.path.join(models_dir, "textual_inversion")
rife_dir = os.path.join(models_dir, "RIFE")

device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"

def get_64x_num(num):
    return math.ceil(num / 64) * 64

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
                "repo_id": ("STRING",{
                    "default": "philz1337x/flat2DAnimerge_v45Sharp"
                }),
                "filename":("STRING",{
                    "default": "flat2DAnimerge_v45Sharp.safetensors"
                })
            },
            "optional":{
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            }
        }
    RETURN_TYPES = ("SD_MODEL_PATH",)
    FUNCTION = "load_checkpoint"

    CATEGORY = "AIFSH_DiffSynth-Studio"

    def load_checkpoint(self,repo_id,filename,ckpt_name=None):
        if ckpt_name:
            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        else:
        # download stable_diffusion from hf
            ckpt_path = hf_hub_download(repo_id=repo_id,
                            filename=filename,
                            local_dir=folder_paths.folder_names_and_paths["checkpoints"][0][0])
            
        return (ckpt_path,)

class ControlNetPathLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model_id": ([
                "canny", "depth", "softedge", "lineart", "lineart_anime", "openpose", "tile"
            ],),
            "scale":("FLOAT",{
                "default": 0.5
            })
            },
            "optional":{
                "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
            }
            }

    RETURN_TYPES = ("ControlNetConfigUnit",)
    FUNCTION = "load_controlnet"

    CATEGORY = "AIFSH_DiffSynth-Studio"

    def load_controlnet(self,model_id,scale,control_net_name=None):
        if model_id in ["canny","softedge","lineart","openpose"]:
            filename = f"control_v11p_sd15_{model_id}.pth"
        elif model_id == "tile":
            filename = "control_v11f1e_sd15_tile.pth"
        elif model_id == "depth":
            filename = "control_v11f1p_sd15_depth.pth"
        elif model_id == "lineart_anime":
            filename = "control_v11p_sd15s2_lineart_anime.pth"
        
        if control_net_name:
            controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
            assert filename == control_net_name, f"{model_id} dismatch with {control_net_name}"
        else:
            controlnet_path = hf_hub_download(repo_id="lllyasviel/ControlNet-v1-1",
        
                                             filename=filename,local_dir=folder_paths.folder_names_and_paths["controlnet"][0][0])
        out_dict = {
            "model":ControlNetConfigUnit(
                    processor_id=model_id,
                    model_path=controlnet_path,
                    scale=scale
                ),
            "path":controlnet_path
        }
        return (out_dict,)

class DiffutoonNode:
    def __init__(self):
        try:
            # AnimateDiff
            hf_hub_download(repo_id="guoyww/animatediff",filename="mm_sd_v15_v2.ckpt",local_dir=animatediff_dir)
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
                    "default": 4
                }),
                "animatediff_stride":("INT",{
                    "default": 2
                }),
                "vram_limit_level":("INT",{
                    "default": 0
                }),
            },
            "optional":{
                "controlnet1":("ControlNetConfigUnit",),
                "controlnet2":("ControlNetConfigUnit",),
                "controlnet3":("ControlNetConfigUnit",),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "maketoon"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_DiffSynth-Studio"

    def maketoon(self,source_video_path,sd_model_path,postive_prompt,negative_prompt,start,length,seed,
                 cfg_scale,num_inference_steps,animatediff_batch_size,animatediff_stride,
                 vram_limit_level,controlnet1=None,controlnet2=None,controlnet3=None,):
        # load models
        model_manager = ModelManager(torch_dtype=torch.float16, device=device)
        shutil.rmtree(os.path.join(textual_inversion_dir,".huggingface"),ignore_errors=True)
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
        org_w, org_h = video.shape()
        height, width = (1024,get_64x_num(1024*org_w/org_h)) if org_h > org_w else (get_64x_num(1024*org_h/org_w),1024)
        print(f"orginal size: {org_w}X{org_h} resize to: {height}X{width}")
        video.set_shape(height,width)

        video_meta_data = video.data.reader.get_meta_data()
        fps = round(video_meta_data['fps'])
        duration = round(video_meta_data['duration'])
        print(f"orginal fps: {fps} duration: {duration}")
        assert start < duration and start + length < duration
        if length == -1:
            input_video = [video[i] for i in range(start*fps, len(video))]
        else:
            input_video = [video[i] for i in range(start*fps, (start+length)*fps)]
        print(f"{len(input_video)} frame will be to shade")
        # Toon shading (20G VRAM)
        torch.manual_seed(seed)
        output_video = pipe(
            prompt=postive_prompt,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale, clip_skip=2,
            controlnet_frames=input_video, num_frames=len(input_video),
            num_inference_steps=num_inference_steps, height=height, width=width,
            animatediff_batch_size=animatediff_batch_size, animatediff_stride=animatediff_stride,
            vram_limit_level=vram_limit_level,
        )
        output_video = smoother(output_video)

        # Save video
        outfile = os.path.join(output_dir,f'{time.time_ns()}_shaded' + os.path.basename(source_video_path))
        save_video(output_video, outfile, fps=fps)
        return (outfile,)