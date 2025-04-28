# SD-CN Animation Parameters Guide

## Overview

SD-CN Animation is a text-to-video animation system that uses Stable Diffusion and ControlNet to generate coherent video sequences from a single initial frame. The system predicts optical flow between frames and uses a two-pass diffusion approach to create smooth animations.

## Core Parameters

### Basic Configuration

- **num_frames**: Total number of frames to generate in the animation
- **sampling_steps**: Number of diffusion steps per frame
- **cfg**: Classifier-free guidance scale (controls prompt adherence)
- **sampler_name**: Diffusion sampler to use
- **scheduler**: Diffusion scheduler
- **seed**: Random seed for reproducibility

### Diffusion Strength Parameters

- **processing_strength**: Controls the strength of the first diffusion pass. Higher values (0.8-1.0) give the model more freedom to change the frame and fix warping artifacts. Lower values (0.3-0.6) preserve more details from the predicted frame.

- **fix_frame_strength**: Controls the strength of the second diffusion pass. Lower values (0.1-0.2) maintain frame-to-frame consistency. Higher values (0.3-0.5) allow more variation and detail refinement.

### Occlusion Mask Parameters

Parameters are listed in the exact order they appear in the node:

- **occlusion_mask_blur**: Controls the Gaussian blur applied to the occlusion mask. Higher values (10-15) create softer transitions that blend more naturally. Lower values (1-5) create more defined masks but may cause visible seams.

- **occlusion_mask_multiplier**: Initial multiplier applied to the raw occlusion mask. The original code uses a value of 10. Higher values detect more subtle motion, while lower values (3-5) focus only on significant movement.

- **occlusion_flow_multiplier**: Direct multiplier for optical flow values. Values below 1.0 (0.6-0.8) reduce sensitivity to fast movement. Values above 1.0 amplify detected motion but may increase distortions.

- **occlusion_difo_multiplier**: Controls how quickly flow influence diminishes with distance. Higher values (1.5-2.0) localize motion effects, reducing smearing artifacts. Lower values (0.5-1.0) create smoother transitions but may cause warping.

- **occlusion_difs_multiplier**: Final multiplier applied to the occlusion mask after processing. The original code uses a value of 25, but testing shows values of 1-3 work better for most scenes. Higher values can cause extreme warping.

### ControlNet Configuration

- **cn_frame_send**: Determines which frame is sent to ControlNet:
  - **None**: No frame is sent
  - **Current Frame**: Current processed frame is sent
  - **Previous Frame**: Previous completed frame is sent

- **controlnet_strength**: Controls how strongly ControlNet influences the output
- **controlnet_start_percent** & **controlnet_end_percent**: When in the diffusion process ControlNet is applied

## Example Configurations

### Balanced Animation (General Purpose)
```
processing_strength: 0.85
fix_frame_strength: 0.15
occlusion_mask_blur: 5
occlusion_mask_multiplier: 5
occlusion_flow_multiplier: 1.0
occlusion_difo_multiplier: 1.0
occlusion_difs_multiplier: 2
controlnet_strength: 0.8
cn_frame_send: Previous Frame
```

### Handling Heavy Warping
```
processing_strength: 1.0
fix_frame_strength: 0.15
occlusion_mask_blur: 15
occlusion_mask_multiplier: 3
occlusion_flow_multiplier: 0.7
occlusion_difo_multiplier: 1.8
occlusion_difs_multiplier: 1.5
controlnet_strength: 0.6
cn_frame_send: Previous Frame
```

### Subtle Movement (e.g., talking head)
```
processing_strength: 0.7
fix_frame_strength: 0.1
occlusion_mask_blur: 3
occlusion_mask_multiplier: 5
occlusion_flow_multiplier: 1.0
occlusion_difo_multiplier: 1.0
occlusion_difs_multiplier: 1.0
controlnet_strength: 0.9
cn_frame_send: Previous Frame
```

### Creative Animation (more variation)
```
processing_strength: 0.9
fix_frame_strength: 0.3
occlusion_mask_blur: 7
occlusion_mask_multiplier: 5
occlusion_flow_multiplier: 1.2
occlusion_difo_multiplier: 0.8
occlusion_difs_multiplier: 2.0
controlnet_strength: 0.5
cn_frame_send: Current Frame
```

## Troubleshooting

### Warping and Distortion Issues
- **Problem**: Excessive warping that diffusion can't fix
- **Solution**: 
  - Increase `processing_strength` to 0.9-1.0
  - Keep `fix_frame_strength` low (0.1-0.15)
  - Increase `occlusion_mask_blur` to 10-15
  - Lower `occlusion_mask_multiplier` to 3-4
  - Reduce `occlusion_flow_multiplier` to 0.7
  - Increase `occlusion_difo_multiplier` to 1.5-2.0
  - Keep `occlusion_difs_multiplier` low (1-1.5)

### Flickering Between Frames
- **Problem**: Content changes too much from frame to frame
- **Solution**: 
  - Lower `fix_frame_strength` to 0.1
  - Increase `controlnet_strength` to 0.9
  - Set `cn_frame_send` to "Previous Frame"

### Loss of Detail in Motion Areas
- **Problem**: Moving areas become blurry or lose important details
- **Solution**: 
  - Increase `processing_strength` to 0.9
  - Use moderate `occlusion_mask_multiplier` (4-6)
  - Reduce `occlusion_mask_blur` to preserve edge detail
  - Increase sampling steps to 20-25

### Too Static Animation
- **Problem**: Not enough movement or variation between frames
- **Solution**: 
  - Increase `processing_strength` to 0.9
  - Increase `fix_frame_strength` to 0.2-0.3
  - Moderately increase `occlusion_mask_multiplier` and `occlusion_difs_multiplier`
  - Reduce `controlnet_strength` slightly

## Advanced Tips

1. **For smoother transitions**: Increase `occlusion_mask_blur` and use "Previous Frame" for ControlNet input

2. **For better detail preservation**: Use more sampling steps (20-30) and higher CFG values (7-8)

3. **For faster generation**: Reduce sampling steps to 15-20 and use "euler" or "dpm_fast" samplers

4. **For character consistency**: Use a strong character LoRA and set `controlnet_strength` to 0.8-0.9

5. **For complex motion**: Start with a higher `processing_strength` (0.9-1.0) and carefully balance occlusion mask parameters