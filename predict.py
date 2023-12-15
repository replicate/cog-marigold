import os
import time
import shutil
import subprocess
from glob import glob
from typing import List

import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset
from cog import BasePredictor, Input, Path

from src.util.seed_all import seed_all
from src.util.batchsize import find_batch_size
from src.util.ensemble import ensemble_depths
from src.model.marigold_pipeline import MarigoldPipeline
from src.util.image_util import chw2hwc, colorize_depth_maps, resize_max_res


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_url = "https://weights.replicate.delivery/default/marigold/checkpoint.tar"
        if not os.path.exists("/src/checkpoint"):
            print("Downloading checkpoint")
            try:
                output = subprocess.check_output(["pget", "-x", ckpt_url, "/src/tmp"])
                os.rename("/src/tmp/", "/src/checkpoint")
            except subprocess.CalledProcessError as e:
                raise e
    
        # load model
        self.model = MarigoldPipeline.from_pretrained("/src/checkpoint", enable_xformers=True)
        self.model.to(self.device)
        self.model.unet.eval()

    def predict(
        self,
        image: Path = Input(description="Input image, use an RGB image for optimal results."),
        resize_input: bool = Input(description="Resize the original input resolution to max resolution.", default=True),
        num_infer: int = Input(
            ge=1, le=20, default=10,
            description="Number of inferences to be ensembled, a higher number gives better results but runs slower."
        ),
        denoise_steps: int = Input(
            ge=1, le=50, default=10,
            description="Inference denoising steps, more steps results in higher accuracy but slower inference speed."
        ),
        regularizer_strength: float = Input(
            ge=0.0, le=1, default=0.02,
            description="Ensembling parameter, weight of optimization regularizer.",
        ),
        reduction_method: str = Input(
            choices=["mean", "median"], default="median",
            description="Ensembling parameter, method to merge aligned depth maps."
        ),
        max_iter: int = Input(ge=1, le=20, default=5, description="Ensembling parameter, max optimization iterations."),
        seed: int = Input(description="Seed for reproducibility, set to random if left as None.", default=None),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int(time.time())
            
        seed_all(seed)

        resize_back = True if resize_input else False
        n_repeat = num_infer
        merging_max_res = None

        # Read input image
        input_image = Image.open(str(image))
        input_size = input_image.size

        # Resize image
        if resize_input:
            input_image = resize_max_res(input_image, max_edge_resolution=768)

        # Convert the image to RGB, to 1.remove the alpha channel 2.convert B&W to 3-channel
        input_image = input_image.convert("RGB")
        image = np.asarray(input_image)

        # Normalize rgb values
        rgb = np.transpose(image, (2, 0, 1))  # [H, W, rgb] -> [rgb, H, W]
        rgb_norm = rgb / 255.0
        rgb_norm = torch.from_numpy(rgb_norm).float()
        rgb_norm = rgb_norm.to(self.device)
        assert rgb_norm.min() >= 0.0 and rgb_norm.max() <= 1.0

        # Batch repeated input image
        duplicated_rgb = torch.stack([rgb_norm] * n_repeat)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        _bs = find_batch_size(n_repeat=n_repeat, input_res=max(rgb_norm.shape[1:]))

        single_rgb_loader = DataLoader(single_rgb_dataset, batch_size=_bs, shuffle=False)
            
        # inference
        with torch.no_grad():
            # Predict depth maps (batched)
            depth_pred_ls = []
            for batch in tqdm(single_rgb_loader, desc="multiple inference", leave=False):
                (batched_img,) = batch
                depth_pred_raw = self.model.forward(
                    batched_img,
                    num_inference_steps=denoise_steps,
                    init_depth_latent=None,
                    show_pbar=True
                )
                # clip prediction
                depth_pred_raw = torch.clip(depth_pred_raw, -1.0, 1.0)
                # shift to [0, 1]
                depth_pred_raw = depth_pred_raw * 2.0 - 1.0
                depth_pred_ls.append(depth_pred_raw.detach().clone())

            depth_preds = torch.concat(depth_pred_ls, axis=0).squeeze()
            torch.cuda.empty_cache()

            # Test-time ensembling
            if n_repeat > 1:
                depth_pred, pred_uncert = ensemble_depths(
                    depth_preds,
                    regularizer_strength=regularizer_strength,
                    max_iter=max_iter,
                    tol=1e-3,
                    reduction=reduction_method,
                    max_res=merging_max_res,
                    device=self.device,
                )
            else:
                depth_pred = depth_preds
            
            # Convert to numpy for saving
            depth_pred = depth_pred.cpu().numpy()

            # Resize back to original resolution
            if resize_back:
                pred_img = Image.fromarray(depth_pred)
                pred_img = pred_img.resize(input_size)
                depth_pred = np.asarray(pred_img)

            # Save as 16-bit uint png
            bw_path = "/tmp/depth_bw.png"
            # scale prediction to [0, 1]
            min_d = np.min(depth_pred)
            max_d = np.max(depth_pred)
            depth_to_save = (depth_pred - min_d) / (max_d - min_d)
            depth_to_save = (depth_to_save * 65535.0).astype(np.uint16)
            Image.fromarray(depth_to_save).save(bw_path, mode="I;16")

            # Colorize
            percentile = 0.03
            min_depth_pct = np.percentile(depth_pred, percentile)
            max_depth_pct = np.percentile(depth_pred, 100 - percentile)
            color_path = "/tmp/depth_colored.png"
            
            # [3, H, W] - values in (0, 1)
            depth_colored = colorize_depth_maps(
                depth_pred, min_depth_pct, max_depth_pct, cmap="Spectral"
            ).squeeze()  
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            Image.fromarray(depth_colored_hwc).save(color_path)

        return [Path(bw_path), Path(color_path)]
