# Cog wrapper for Marigold

A Cog wrapper for Marigold, a diffusion model and associated fine-tuning protocol for monocular depth estimation. See the original [repository](https://github.com/prs-eth/marigold), [paper](https://arxiv.org/abs/2312.02145) and [Replicate demo](https://replicate.com/adirik/marigold) for details.

## API Usage

You need to have Cog and Docker installed to run this model locally. Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of Marigold to [Replicate](https://replicate.com). To use the model, simply provide the image (ideally RGB or grayscale) you would like to perform depth estimation for. The API returns two depth map images - one grayscale and one spectral.

To build the docker image with cog and run a prediction:
```bash
cog predict -i image=@examples/files_bee.jpg
```

To start a server and send requests to your locally or remotely deployed API:
```bash
cog run -p 5000 python -m cog.server.http
```

Input parameters are as follows:  
- **image:** RGB or grayscale input image for the model, use an RGB image for best results.  
- **resize_input:** whether to resize the input image to max resolution of 768 x 768 pixels, default to `True`.  
- **num_infer:** number of inferences to be performed. if >1, multiple depth predictions are ensembled. A higher number yields better results but runs slower.   
- **denoise_steps:** number of inference denoising steps, more steps results in higher accuracy but slower inference speed.  
- **regularizer_strength:** ensembling parameter, weight of optimization regularizer.  
- **reduction_method:** ensembling parameter, method to merge aligned depth maps. Choose between `["mean", "median"]`.  
- **max_iter:** ensembling parameter, max number of optimization iterations.   
- **seed:** (optional) seed for reproducibility, set to random if left as `None`.   

## References 
```
@misc{ke2023repurposing,
      title={Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation}, 
      author={Bingxin Ke and Anton Obukhov and Shengyu Huang and Nando Metzger and Rodrigo Caye Daudt and Konrad Schindler},
      year={2023},
      eprint={2312.02145},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```