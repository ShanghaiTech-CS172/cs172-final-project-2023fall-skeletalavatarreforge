import pickle
import rich
import cv2
import numpy as np
import pandas as pd
import torch
import time
import re
import os
import os.path
import json
import torchvision
import imageio
import warnings
from matplotlib import pyplot as plt
from plyfile import PlyData, PlyElement
from gaussian_renderer import GaussianModel
from scene import Scene
from tqdm import tqdm
from smpl.smpl_numpy import SMPL
from gaussian_renderer import render as _render
from dataclasses import dataclass
from scene.cameras import smpl_to_cuda
from PIL import Image
from typing import Optional

from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, getProjectionMatrix, getProjectionMatrix_refine
from utils.save_utils import save_ply, save_img


smpl_model = SMPL(sex='neutral', model_dir='assets/SMPL_NEUTRAL_renderpeople.pkl')

big_pose_smpl_param = {}
big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
big_pose_smpl_param['Th'] = np.zeros((1,3)).astype(np.float32)
big_pose_smpl_param['shapes'] = np.zeros((1,10)).astype(np.float32)
big_pose_smpl_param['poses'] = np.zeros((1,72)).astype(np.float32)
big_pose_smpl_param['poses'][0, 5] = 45/180*np.array(np.pi)
big_pose_smpl_param['poses'][0, 8] = -45/180*np.array(np.pi)
big_pose_smpl_param['poses'][0, 23] = -30/180*np.array(np.pi)
big_pose_smpl_param['poses'][0, 26] = 30/180*np.array(np.pi)

big_pose_xyz, _ = smpl_model(big_pose_smpl_param['poses'], big_pose_smpl_param['shapes'].reshape(-1))
big_pose_xyz = (np.matmul(big_pose_xyz, big_pose_smpl_param['R'].transpose()) + big_pose_smpl_param['Th']).astype(np.float32)
big_pose_smpl_param = smpl_to_cuda(big_pose_smpl_param, 'cuda')


@dataclass
class DatasetConfig:

    actor_gender = 'neutral'
    data_device = 'cuda'
    eval = True
    exp_name = 'zju_mocap_refine/my_377_100_pose_correction_lbs_offset_split_clone_merge_prune'
    images = 'images'
    model_path = 'output/zju_mocap_refine/my_377_100_pose_correction_lbs_offset_split_clone_merge_prune'
    motion_offset_flag = True
    resolution = -1
    sh_degree = 3
    smpl_type = 'smpl'
    source_path = '/home/user/GauHuman/data/zju_mocap_refine/my_377'
    white_background = False


@dataclass
class PipelineConfig:

    convert_SHs_python = False
    compute_cov3D_python = True
    debug = False


@dataclass
class RenderOption:

    # camera parameters
    FoVx: float
    FoVy: float
    image_height: int
    image_width: int
    world_view_transform: np.ndarray
    full_proj_transform: np.ndarray
    camera_center: np.ndarray

    # pose parameters
    smpl_param: dict


RenderOption.big_pose_smpl_param = smpl_to_cuda(big_pose_smpl_param, 'cuda')
RenderOption.big_pose_world_vertex = torch.tensor(big_pose_xyz, dtype=torch.float32, device="cuda")

def create_render_options(image_path, cam, image_scaling=0.5):
    
    path, colmap_id, pose_id = re.search(r"(/home/user/GauHuman/data/zju_mocap_refine/my_\d+)/images/(\d+)/(\d+)\.jpg", image_path).groups()
    print(path, colmap_id, pose_id)

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255

    msk_path = image_path.replace('images', 'mask').replace('jpg', 'png')
    msk = cv2.cvtColor(cv2.imread(msk_path), cv2.COLOR_BGR2RGB)
    msk = (msk != 0).astype(np.uint8)


    K = np.array(cam['K'])
    D = np.array(cam['D'])
    R = np.array(cam['R'])
    T = np.array(cam['T']) / 1000.

    image = cv2.undistort(image, K, D)
    msk = cv2.undistort(msk, K, D)

    image[msk == 0] = 0

    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    w2c = np.eye(4)
    w2c[:3,:3] = R
    w2c[:3,3:4] = T

    # get the world-to-camera transform and set R, T
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]

    # Reduce the image resolution by ratio, then remove the back ground
    ratio = image_scaling
    if ratio != 1.:
        H, W = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        K[:2] = K[:2] * ratio

    image = Image.fromarray(np.array(image*255.0, dtype=np.byte), "RGB")

    focalX = K[0,0]
    focalY = K[1,1]
    FovX = focal2fov(focalX, image.size[0])
    FovY = focal2fov(focalY, image.size[1])

    # load smpl data 
    i = int(os.path.basename(image_path)[:-4])

    smpl_param_path = os.path.join(path, f'smpl_params/{i}.npy')
    smpl_param = np.load(smpl_param_path, allow_pickle=True).item()
    Rh = smpl_param['Rh']
    smpl_param['R'] = cv2.Rodrigues(Rh)[0].astype(np.float32)
    smpl_param['Th'] = smpl_param['Th'].astype(np.float32)
    smpl_param['shapes'] = smpl_param['shapes'].astype(np.float32)
    smpl_param['poses'] = smpl_param['poses'].astype(np.float32)


    zfar = 1000 #100.0
    znear = 0.001 #0.01

    trans = np.array([0, 0, 0])
    scale = 1.0
    world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
    projection_matrix = getProjectionMatrix_refine(torch.tensor(K, device='cuda', dtype=torch.float32), image.size[1], image.size[0], znear, zfar).transpose(0, 1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    smpl_param = smpl_to_cuda(smpl_param, 'cuda')

    return RenderOption(FoVx=FovX, FoVy=FovY, 
                        image_height=image.size[1], image_width=image.size[0], 
                        world_view_transform=world_view_transform,
                        full_proj_transform=full_proj_transform, 
                        camera_center=camera_center, 
                        smpl_param=smpl_param)

def render(pose: np.ndarray, output_filename: Optional[str] = None, white_back_groud = False) -> np.ndarray:
    '''
    Parameters
    ----------
    pose: torch.Tensor
        (1, 72) SMPL pose parameters, where 72 = (3 (global) + 23 (skel)) * 3 (axis-angle)
    output_filename: str
        output filename
    '''
    assert pose.shape == (1, 72)
    render_opt.smpl_param['poses'] = torch.tensor(pose, dtype=torch.float32, device='cuda')
    render_res = _render(render_opt, gaussians, pipe=PipelineConfig(), bg_color=torch.tensor([1 if white_back_groud else 0] * 3, dtype=torch.float32, device="cuda"))
    img = render_res['render']
    if output_filename is not None:
        save_img(img, output_filename)
    return img.detach().cpu().numpy().transpose((1, 2, 0))
    

path = '/home/user/GauHuman/data/zju_mocap_refine/my_377'

ann_file = os.path.join(path, 'annots.npy')
annots = np.load(ann_file, allow_pickle=True).item()
cams = annots['cams']
for c in 'KDRT':
    cams[c] = np.array(cams[c])

    
dataset = DatasetConfig()
gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender)
scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")


render_opt = create_render_options(
    image_path='/home/user/GauHuman/data/zju_mocap_refine/my_377/images/04/000000.jpg',
    cam=dict(
        K=cams['K'][4],
        D=cams['D'][4],
        R=cams['R'][4],
        T=cams['T'][4],
    )
)

