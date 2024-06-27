# ComfyUI-DiffSynth-Studio
make [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) avialbe in ComfyUI
<div>
  <figure>
  <img alt='webpage' src="web.png?raw=true" width="600px"/>
  <figure>
</div>

## how to use
test on py3.10，2080ti 11gb，torch==2.3.0+cu121

make sure `ffmpeg` is worked in your commandline
for Linux
```
apt update
apt install ffmpeg
```
for Windows,you can install `ffmpeg` by [WingetUI](https://github.com/marticliment/WingetUI) automatically

then!
```
# in ComfyUI/custom_nodes
git clone https://github.com/AIFSH/ComfyUI-DiffSynth-Studio.git
cd ComfyUI-DiffSynth-Studio
pip install -r requirements.txt
```
weights will be downloaded from huggingface or model scope

## Tutorial
- [Demo for Diffutoon](https://b23.tv/z7hEXlX)[DiffSynth-Studio!ComfyUI插件之Diffutoon节点-哔哩哔哩](https://b23.tv/z7hEXlX)
- [Demo for ExVideo](https://b23.tv/z7hEXlX)[DiffSynth-Studio!ComfyUI插件之ExVideo节点-哔哩哔哩](https://b23.tv/z7hEXlX)

## Nodes Detail and Workflow
### ExVideo Node
[ExVideo workflow](./workfolws/exvideo_workflow.json)
```
"image":("IMAGE",),
"svd_base_model":("SD_MODEL_PATH",),
"exvideo_model":("SD_MODEL_PATH",),
"num_frames":("INT",{ 
    "default": 128
}),
"fps":("INT",{
    "default": 30
}),
"num_inference_steps":("INT",{
    "default": 50
}),
"if_upscale":("BOOLEAN",{
    "default": True,
}),
"seed": ("INT",{
    "default": 1
})
```
### image synthesis
comming soon
### Diffutoon Node
[Diffutoon workflow](./workfolws/diffutoon_workflow.json)
```
"required":{
    "source_video_path": ("VIDEO",),
    "sd_model_path":("SD_MODEL_PATH",),
    "postive_prompt":("TEXT",),
    "negative_prompt":("TEXT",),
    "start":("INT",{
        "default": 0 ## from which second of your video to be shaded
    }),
    "length":("INT",{
        "default": -1 ## how long you want to shade, -1 name the whole video frames
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
        "default": 32 ## lower it till you can run
    }),
    "animatediff_stride":("INT",{
        "default": 16 ## lower it till you can run
    }),
    "vram_limit_level":("INT",{
        "default": 0 ## meet killed? try to 1
    }),
},
"optional":{
    "controlnet1":("ControlNetConfigUnit",),
    "controlnet2":("ControlNetConfigUnit",),
    "controlnet3":("ControlNetConfigUnit",),
}
```

### Video Stylization

### Chinese Models

## ask for answer as soon as you want
wechat: aifsh_98
need donate if you mand it,
but please feel free to new issue for answering

Windows环境配置太难？可以添加微信：aifsh_98，赞赏获取Windows一键包，当然你也可以提issue等待大佬为你答疑解惑。