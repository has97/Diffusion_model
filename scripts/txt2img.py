import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
import cv2

from ldm.util import instantiate_from_config
import torchvision.transforms as transforms
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    opt = parser.parse_args()


    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")  # TODO: check path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt


    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples=list()
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.n_samples * [""])
            for n in trange(opt.n_iter, desc="Sampling"):
                c = model.get_learned_conditioning(opt.n_samples * [prompt])
                shape = [4, opt.H//8, opt.W//8]
                ################### reading the image here ###################
                img1 = cv2.imread('data/images/image7.jpg',0)
                img1 = cv2.resize(img1,(opt.H//8,opt.W//8))
                img2 = cv2.imread('data/images/image4.jpg',0)
                img2 = cv2.resize(img2,(opt.H//8,opt.W//8))
                img3 = cv2.imread('data/images/image12.jpg',0)
                img3 = cv2.resize(img3,(opt.H//8,opt.W//8))
                img4 = cv2.imread('data/images/image9.jpg',0)
                img4 = cv2.resize(img4,(opt.H//8,opt.W//8))
                ##############################################################
                
                ################## Normalizing the image #######################
                img1 = cv2.normalize(img1, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
                img2 = cv2.normalize(img2, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
                img3 = cv2.normalize(img3, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
                img4 = cv2.normalize(img4, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
                ###############################################################
#                 print(np.max(img))
#                 print(np.min(img))
#                 print(img)
                ######## converting to tensor ##############
                transform = transforms.ToTensor()
                img1 = transform(img1)
                img2 = transform(img2)
                img3 = transform(img3)
                img4 = transform(img4)
                ###########################################
#                 img1 += torch.randn((opt.H//8, opt.W//8))
#                 img2 += torch.randn((opt.H//8, opt.W//8))
#                 img3 += torch.randn((opt.H//8, opt.W//8))
#                 img4 += torch.randn((opt.H//8, opt.W//8))
#                 img1 /=2
#                 img2 /=2
#                 img3 /=2
#                 img4 /=2
                ######## here the input image is rescaled to the size of the input noise ##############
                x_i = torch.zeros((4,4,opt.H//8, opt.W//8),device=device)
                x_i[0] = torch.cat((img1,img1,img1,img1))
                x_i[1] = torch.cat((img2,img2,img2,img2))
                x_i[2] = torch.cat((img3,img3,img3,img3))
                x_i[3] = torch.cat((img4,img4,img4,img4))
                #######################################################################################
#                 print(x_T[0].shape)
#                 print(img[0].shape)
#                 x_T = x_T/255
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
#                                                  x_T = x_T,
                                                 x_i=x_i, # the intermediate imagenet images passed to sampling function.
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                    base_count += 1
                all_samples.append(x_samples_ddim)


    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=opt.n_samples)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.png'))

    print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")
