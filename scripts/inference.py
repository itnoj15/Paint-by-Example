import os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
import clip
from torchvision.transforms import Resize

wm = "Paint-by-Example"
wm_encoder = WatermarkEncoder()
wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def load_model_from_config(config, ckpt, verbose=False):
    # print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    # if "global_step" in pl_sd:
    #     print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    # if len(m) > 0 and verbose:
    #     print("missing keys:")
    #     print(m)
    # if len(u) > 0 and verbose:
    #     print("unexpected keys:")
    #     print(u)

    model.cuda()
    model.eval()
    return model

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def main():

    DDIM_STEPS = 100
    IS_PLMS = True
    IS_FIXED_CODE = False
    DDIM_ETA = 0.0
    H = 512
    W = 512
    C = 4
    F = 8
    N_SAMPLES = 1
    SCALE = 5
    SEED = 245
    CONFIG_PATH = "configs/v1.yaml"
    MODEL_PATH = "checkpoints/model.ckpt"
    PRECISION = "full" # "autocast"
    OUTDIR = "results"

    IMAGE_PATH = "examples/image/99700886_421.png"
    MASK_PATH = "examples/mask/99700886_main.png"
    REFERENCE_PATH = "examples/reference/99700886_main.jpg"

    seed_everything(SEED)

    config = OmegaConf.load(f"{CONFIG_PATH}")
    model = load_model_from_config(config, f"{MODEL_PATH}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device).to(torch.float32)

    if IS_PLMS:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    result_path = OUTDIR
    os.makedirs(result_path, exist_ok=True)

    start_code = None
    if IS_FIXED_CODE:
        start_code = torch.randn([N_SAMPLES, C, H // F, W // F], device=device, dtype=torch.float32)

    precision_scope = autocast if PRECISION=="autocast" else nullcontext

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                filename = os.path.basename(IMAGE_PATH)

                img_p = Image.open(IMAGE_PATH).convert("RGB")
                original_size = img_p.size

                img_p = img_p.resize((W, H), Image.LANCZOS)
                image_tensor = get_tensor()(img_p)
                image_tensor = image_tensor.unsqueeze(0).to(torch.float32).to(device)

                ref_p = Image.open(REFERENCE_PATH).convert("RGB").resize((224, 224), Image.LANCZOS)
                ref_tensor = get_tensor_clip()(ref_p)
                ref_tensor = ref_tensor.unsqueeze(0).to(torch.float32).to(device)

                mask = Image.open(MASK_PATH).convert("L")
                mask = mask.resize((W, H), Image.LANCZOS)  
                mask = np.array(mask)[None, None]
                mask = 1 - mask.astype(np.float32) / 255.0
                mask[mask < 0.5] = 0
                mask[mask >= 0.5] = 1
                mask_tensor = torch.from_numpy(mask).to(torch.float32).to(device)

                inpaint_image = image_tensor * mask_tensor
                test_model_kwargs = {}
                test_model_kwargs['inpaint_mask'] = mask_tensor.to(device)
                test_model_kwargs['inpaint_image'] = inpaint_image.to(device)
                ref_tensor = ref_tensor.to(device)

                uc = None
                if SCALE != 1.0:
                    uc = model.learnable_vector
                c = model.get_learned_conditioning(ref_tensor.to(torch.float32))
                c = model.proj_out(c)

                z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
                z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()

                test_model_kwargs['inpaint_image'] = z_inpaint
                test_model_kwargs['inpaint_mask'] = Resize([z_inpaint.shape[-2], z_inpaint.shape[-1]])(test_model_kwargs['inpaint_mask'])

                shape = [C, H // F, W // F]
                samples_ddim, _ = sampler.sample(S=DDIM_STEPS,
                                                 conditioning=c,
                                                 batch_size=N_SAMPLES,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=SCALE,
                                                 unconditional_conditioning=uc,
                                                 eta=DDIM_ETA,
                                                 x_T=start_code,
                                                 test_model_kwargs=test_model_kwargs)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                x_checked_image = x_samples_ddim
                x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                for i, x_sample in enumerate(x_checked_image_torch):
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img = img.resize(original_size, Image.LANCZOS) 
                    img.save(os.path.join(result_path, filename[:-4] + '_' + str(SEED) + ".png"))

    # print(f"Your samples are ready and waiting for you here: \n{result_path} \nEnjoy.")

if __name__ == "__main__":
    main()
