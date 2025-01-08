import os
import io
import cv2
import json
import numpy
import random
# import trimesh
import matplotlib.pyplot as plotlib
from typing import Optional, List
from torch.utils.data import Dataset
from mmgen.datasets.builder import DATASETS
from mmcv.parallel import DataContainer as DC
from multiprocessing.pool import ThreadPool
import torch

import torch.nn.functional as F

from scipy.ndimage import label

from .parallel_zip import ParallelZipFile as ZipFile




import h5py
import cv2
import numpy as np 
from tqdm import tqdm 
import pdb


def recenter(image, mask, bg_color, border_ratio = 0.2):
    """ recenter an image to leave some empty space at the image border.

    Args:
        image (ndarray): input image, float/uint8 [H, W, 3/4]
        mask (ndarray): alpha mask, bool [H, W]
        border_ratio (float, optional): border ratio, image will be resized to (1 - border_ratio). Defaults to 0.2.

    Returns:
        ndarray: output image, float/uint8 [H, W, 3/4]
    """
    
    H, W, C = image.shape
    size = max(H, W)

    # default to white bg if rgb, but use 0 if rgba
    if C == 3:
        result = np.ones((size, size, C), dtype=np.float32)*255
    else:
        result = np.zeros((size, size, C), dtype=np.float32)
        result[:,:,:3] = bg_color
            
    coords = np.nonzero(mask)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    
    h = x_max - x_min
    w = y_max - y_min
    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(h, w)
    h2 = int(h * scale)
    w2 = int(w * scale)
    x2_min = (size - h2) // 2
    x2_max = x2_min + h2
    y2_min = (size - w2) // 2
    y2_max = y2_min + w2
    result[x2_min:x2_max, y2_min:y2_max] = cv2.resize(image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
    
    return result, x_min, y_min, h2/(x_max-x_min), w2/(y_max-y_min), x2_min, y2_min


def visual_hull_samples(masks, KRT, grid_resolution=64, aabb=(-1.0, 1.0)):
    """ 
    Args:
        masks: (n_images, H, W)
        KRT: (n_images, 3, 4)
        grid_resolution: int
        aabb: (2)
    """
    # create voxel grid coordinates
    grid = np.linspace(aabb[0], aabb[1], grid_resolution) # sample grid_resolution points in interval aabb
    grid = np.meshgrid(grid, grid, grid) # make it 3d grid
    grid_loc = np.stack(grid, axis=-1).reshape(-1, 3) # n_pts, 3

    # project grid locations to the image plane
    grid = np.concatenate([grid_loc, np.ones_like(grid_loc[:, :1])], axis=-1) # n_pts, 4
    grid = grid[None].repeat(masks.shape[0], axis=0) # n_imgs, n_pts, 4
    grid = grid @ KRT.transpose(0, 2, 1)  # (n_imgs, n_pts, 4) @ (n_imgs, 4, 3) -> (n_imgs, n_pts, 3)
    uv = grid[..., :2] / grid[..., 2:] # (n_imgs, n_pts, 2)
    _, H, W = masks.shape[:3]  # n_imgs,H,W
    uv[..., 0] = 2.0 * (uv[..., 0] / (W - 1.0)) - 1.0
    uv[..., 1] = 2.0 * (uv[..., 1] / (H - 1.0)) - 1.0

    uv = torch.from_numpy(uv).float()
    masks = torch.from_numpy(masks)[:, None].squeeze(-1).float()
    samples = F.grid_sample(masks, uv[:, None], align_corners=True, mode='nearest', padding_mode='zeros').squeeze()
    _ind = (samples > 0).all(0) # (n_imgs, n_pts) -> (n_pts)

    # sample points around the grid locations
    grid_samples = grid_loc[_ind] # (n_pts, 2)
    all_samples = grid_samples
    np.random.shuffle(all_samples)
    
    return all_samples

class SMCReader:

    def __init__(self, file_path):
        """Read SenseMocapFile endswith ".smc".

        Args:
            file_path (str):
                Path to an SMC file.
            body_model (nn.Module or dict):
                Only needed for SMPL transformation to device frame
                if nn.Module: a body_model instance
                if dict: a body_model config
        """
        self.smc = h5py.File(file_path, 'r')
        self.__calibration_dict__ = None
        self.__kinect_calib_dict__ = None 
        self.__available_keys__ = list(self.smc.keys())
        
        self.actor_info = None 
        if hasattr(self.smc, 'attrs') and len(self.smc.attrs.keys()) > 0:
            self.actor_info = dict(
                id=self.smc.attrs['actor_id'],
                perf_id=self.smc.attrs['performance_id'],
                age=self.smc.attrs['age'],
                gender=self.smc.attrs['gender'],
                height=self.smc.attrs['height'],
                weight=self.smc.attrs['weight'],
                # ethnicity=self.smc.attrs['ethnicity'],
            )

        self.Camera_5mp_info = None 
        if 'Camera_5mp' in self.smc:
            self.Camera_5mp_info = dict(
                num_device=self.smc['Camera_5mp'].attrs['num_device'],
                num_frame=self.smc['Camera_5mp'].attrs['num_frame'],
                resolution=self.smc['Camera_5mp'].attrs['resolution'],
            )
        self.Camera_12mp_info = None 
        if 'Camera_12mp' in self.smc:
            self.Camera_12mp_info = dict(
                num_device=self.smc['Camera_12mp'].attrs['num_device'],
                num_frame=self.smc['Camera_12mp'].attrs['num_frame'],
                resolution=self.smc['Camera_12mp'].attrs['resolution'],
            )
        self.Kinect_info = None
        if 'Kinect' in self.smc:
            self.Kinect_info=dict(
                num_device=self.smc['Kinect'].attrs['num_device'],
                num_frame=self.smc['Kinect'].attrs['num_frame'],
                resolution=self.smc['Kinect'].attrs['resolution'],
            )

    def get_available_keys(self):
        return self.__available_keys__ 

    def get_actor_info(self):
        return self.actor_info
    
    def get_Camera_12mp_info(self):
        return self.Camera_12mp_info

    def get_Camera_5mp_info(self):
        return self.Camera_5mp_info
    
    def get_Kinect_info(self):
        return self.Kinect_info
    
    ### RGB Camera Calibration
    def get_Calibration_all(self):
        """Get calibration matrix of all cameras and save it in self
        
        Args:
            None

        Returns:
            Dictionary of calibration matrixs of all matrixs.
              dict( 
                Camera_Parameter: Camera_id : Matrix_type : value
              )
            Notice:
                Camera_id(str) in {'Camera_5mp': '0'~'47',  'Camera_12mp':'48'~'60'}
                Matrix_type in ['D', 'K', 'RT', 'Color_Calibration'] 
        """  
        if not 'Camera_Parameter' in self.smc:
            print("=== no key: Camera_Parameter.\nplease check available keys!")
            return None  

        if self.__calibration_dict__ is not None:
            return self.__calibration_dict__

        self.__calibration_dict__ = dict()
        for ci in self.smc['Camera_Parameter'].keys():
            self.__calibration_dict__.setdefault(ci,dict())
            for mt in ['D', 'K', 'RT', 'Color_Calibration'] :
                self.__calibration_dict__[ci][mt] = \
                    self.smc['Camera_Parameter'][ci][mt][()]
        return self.__calibration_dict__

    def get_Calibration(self, Camera_id):
        """Get calibration matrixs of a certain camera by its type and id 

        Args:
            Camera_id (int/str of a number):
                Camera_id(str) in {'Camera_5mp': '0'~'47',  
                    'Camera_12mp':'48'~'60'}
        Returns:
            Dictionary of calibration matrixs.
                ['D', 'K', 'RT', 'Color_Calibration'] 
        """
        if not 'Camera_Parameter' in self.smc:
            print("=== no key: Camera_Parameter.\nplease check available keys!")
            return None  

        rs = dict()
        for k in ['D', 'K', 'RT', 'Color_Calibration'] :
            rs[k] = self.smc['Camera_Parameter'][f'{int(Camera_id):02d}'][k][()]
        return rs

    ### Kinect Camera Calibration
    def get_Kinect_Calibration_all(self):
        """Get calibration matrix of all kinect cameras and save it in self
        
        Args:
            None

        Returns:
            Dictionary of calibration matrixs of all matrixs.
              dict( 
                Camera_group: Camera_id : Matrix_type : value
              )
            Notice:
                Camera_group(str) in ['Kinect']
                Camera_id(str) in {'Kinect': '0'~'7'}
                Matrix_type in ['D', 'K', 'RT'] 
        """  
        if not 'Calibration' in self.smc:
            print("=== no key: Calibration.\nplease check available keys!")
            return None  

        if self.__kinect_calib_dict__ is not None:
            return self.__kinect_calib_dict__

        self.__kinect_calib_dict__ = dict()
        for cg in ['Kinect']:
            self.__kinect_calib_dict__.setdefault(cg,dict())
            for ci in self.smc['Calibration'][cg].keys():
                self.__kinect_calib_dict__[cg].setdefault(ci,dict())
                for mt in ['D', 'K', 'RT'] :
                    self.__kinect_calib_dict__[cg][ci][mt] = \
                        self.smc['Calibration'][cg][ci][mt][()]
        return self.__kinect_calib_dict__

    def get_kinect_Calibration(self, Camera_id):
        """Get calibration matrixs of a certain kinect camera by its type and id 

        Args:
            Camera_group (str):
                Camera_group in ['Kinect'].
            Camera_id (int/str of a number):
                CameraID(str) in {'Kinect': '0'~'7'}
        Returns:
            Dictionary of calibration matrixs.
                ['D', 'K', 'RT'] 
        """  
        if not 'Calibration' in self.smc:
            print("=== no key: Calibration.\nplease check available keys!")
            return None 

        Camera_id = f'{int(Camera_id):02d}'
        assert(Camera_id in self.smc['Calibration']["Kinect"].keys())
        rs = dict()
        for k in ['D', 'K', 'RT']:
            rs[k] = self.smc['Calibration']["Kinect"][Camera_id][k][()]
        return rs

    ### RGB image
    def __read_color_from_bytes__(self, color_array):
        """Decode an RGB image from an encoded byte array."""
        return cv2.imdecode(color_array, cv2.IMREAD_COLOR)

    def get_mask(self, Camera_id, Frame_id=None,disable_tqdm=True):
        """Get mask from Camera_id, Frame_id

        Args:
            Camera_id (int/str of a number):
                Camera_id (str) in 
                    {'Camera_5mp': '0'~'47',  
                    'Camera_12mp':'48'~'60',
                    'Kinect': '0'~'7'}
            Frame_id a.(int/str of a number): '0' ~ 'num_frame'
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence 
        Returns:
            a single img :
              'color': HWC in bgr (uint8)
              'mask' : HW (uint8)
              'depth': HW (uint16)
        """ 
        if not 'Mask' in self.smc:
            print("=== no key: Mask.\nplease check available keys!")
            return None  

        Camera_id = str(Camera_id)

        assert(isinstance(Frame_id,(list,int, str, type(None))))
        if isinstance(Frame_id, (str,int)):
            Frame_id = str(Frame_id)
            assert(Frame_id in self.smc['Mask'][Camera_id]['mask'].keys())
            img_byte = self.smc['Mask'][Camera_id]['mask'][Frame_id][()]
            img_color = self.__read_color_from_bytes__(img_byte)
            img_color = np.max(img_color,2)
            return img_color           
        else:
            if Frame_id is None:
                Frame_id_list =sorted([int(l) for l in self.smc['Mask'][Camera_id]['mask'].keys()])
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in  tqdm.tqdm(Frame_id_list, disable=disable_tqdm):
                rs.append(self.get_mask(Camera_id,fi))
            return np.stack(rs,axis=0)

    def get_img(self, Camera_group, Camera_id, Image_type,Frame_id=None,disable_tqdm=True):
        """Get image its Camera_group, Camera_id, Image_type and Frame_id

        Args:
            Camera_group (str):
                Camera_group in ['Camera_12mp', 'Camera_5mp','Kinect'].
            Camera_id (int/str of a number):
                CameraID (str) in 
                    {'Camera_5mp': '0'~'47',  
                    'Camera_12mp':'48'~'60',
                    'Kinect': '0'~'7'}
            Image_type(str) in 
                    {'Camera_5mp': ['color'],  
                    'Camera_12mp': ['color'],
                    'Kinect': ['depth', 'mask']}
            Frame_id a.(int/str of a number): '0' ~ 'num_frame'('149') 
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence 
        Returns:
            a single img :
              'color': HWC in bgr (uint8)
              'mask' : HW (uint8)
              'depth': HW (uint16)
        """ 
        if not Camera_group in self.smc:
            print("=== no key: %s.\nplease check available keys!" % Camera_group)
            return None

        assert(Camera_group in ['Camera_12mp', 'Camera_5mp','Kinect'])
        Camera_id = str(Camera_id)
        assert(Camera_id in self.smc[Camera_group].keys())
        assert(Image_type in self.smc[Camera_group][Camera_id].keys())
        assert(isinstance(Frame_id,(list,int, str, type(None))))
        if isinstance(Frame_id, (str,int)):
            Frame_id = str(Frame_id)
            assert(Frame_id in self.smc[Camera_group][Camera_id][Image_type].keys())
            if Image_type in ['color']:
                img_byte = self.smc[Camera_group][Camera_id][Image_type][Frame_id][()]
                img_color = self.__read_color_from_bytes__(img_byte)
            if Image_type == 'mask':
                img_byte = self.smc[Camera_group][Camera_id][Image_type][Frame_id][()]
                img_color = self.__read_color_from_bytes__(img_byte)
                img_color = np.max(img_color,2)
            if Image_type == 'depth':
                img_color = self.smc[Camera_group][Camera_id][Image_type][Frame_id][()]
            return img_color           
        else:
            if Frame_id is None:
                Frame_id_list =sorted([int(l) for l in self.smc[Camera_group][Camera_id][Image_type].keys()])
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in tqdm(Frame_id_list, disable=disable_tqdm):
                rs.append(self.get_img(Camera_group, Camera_id, Image_type,fi))
            return np.stack(rs,axis=0)
    
    ###Keypoints2d
    def get_Keypoints2d(self, Camera_id, Frame_id=None):
        """Get keypoint2D by its Camera_group, Camera_id and Frame_id

        Args:
            Camera_id (int/str of a number):
                CameraID (str) in 
                    {'Camera_5mp': '0'~'47',  
                    'Camera_12mp':'48'~'60',}
            Frame_id a.(int/str of a number): '0' ~ 'num_frame-1'('149') 
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence 
        Returns:
            a single img :
              'color': HWC in bgr (uint8)
              'mask' : HW (uint8)
              'depth': HW (uint16)
        """ 
        if not 'Keypoints_2D' in self.smc:
            print("=== no key: Keypoints_2D.\nplease check available keys!")
            return None 

        Camera_id = f'{int(Camera_id):02d}'
        assert(isinstance(Frame_id,(list,int, str, type(None))))
        if isinstance(Frame_id, (str,int)):
            Frame_id = int(Frame_id)
            return self.smc['Keypoints_2D'][Camera_id][()][Frame_id,:]
        else:
            if Frame_id is None:
                return self.smc['Keypoints_2D'][Camera_id][()]
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in  tqdm.tqdm(Frame_id_list):
                rs.append(self.get_Keypoints2d(Camera_id,fi))
            return np.stack(rs,axis=0)

    ###Keypoints3d
    def get_Keypoints3d(self, Frame_id=None):
        """Get keypoint3D Frame_id, TODO coordinate

        Args:
            Frame_id a.(int/str of a number): '0' ~ 'num_frame-1'('149') 
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence 
        Returns:
            Keypoints3d tensor: np.ndarray of shape ([N], ,3)
        """ 
        if not 'Keypoints_3D' in self.smc:
            print("=== no key: Keypoints_3D.\nplease check available keys!")
            return None 

        if isinstance(Frame_id, (str,int)):
            Frame_id = int(Frame_id)
            return self.smc['Keypoints_3D']["keypoints3d"][Frame_id,:]
        else:
            if Frame_id is None:
                return self.smc['Keypoints_3D']["keypoints3d"]
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in tqdm.tqdm(Frame_id_list):
                rs.append(self.get_Keypoints3d(fi))
            return np.stack(rs,axis=0)

    ###SMPLx
    def get_SMPLx(self, Frame_id=None):
        """Get SMPL (world coordinate) computed by mocap processing pipeline.

        Args:
            Frame_id (int, list or None, optional):
                int: frame id of one selected frame
                list: a list of frame id
                None: all frames will be returned
                Defaults to None.

        Returns:
            dict:
                'global_orient': np.ndarray of shape (N, 3)
                'body_pose': np.ndarray of shape (N, 21, 3)
                'transl': np.ndarray of shape (N, 3)
                'betas': np.ndarray of shape (1, 10)
        """
        if not 'SMPLx' in self.smc:
            print("=== no key: SMPLx.\nplease check available keys!")
            return None 

        t_frame = self.smc['SMPLx']['betas'][()].shape[0]
        print("=== t_frame", self.smc['SMPLx']['betas'][()].shape, flush=True)

        key_arr = ['betas', 'expression', 'fullpose', 'transl']
        for k in key_arr:
            print("=== smplx %s" % k, self.smc['SMPLx'][k][()].shape[0], flush=True)
        
        if Frame_id is None:
            frame_list = range(t_frame)
        elif isinstance(Frame_id, list):
            frame_list = [int(fi) for fi in Frame_id]
        elif isinstance(Frame_id, (int,str)):
            Frame_id = int(Frame_id)
            assert Frame_id < t_frame,\
                f'Invalid frame_index {Frame_id}'
            frame_list = Frame_id
        else:
            raise TypeError('frame_id should be int, list or None.')

        smpl_dict = {}
        for key in ['betas', 'expression', 'fullpose', 'transl']:
            smpl_dict[key] = self.smc['SMPLx'][key][()][frame_list, ...]
        smpl_dict['scale'] = self.smc['SMPLx']['scale'][()]

        return smpl_dict

    def release(self):
        self.smc = None 
        self.__calibration_dict__ = None
        self.__kinect_calib_dict__ = None
        self.__available_keys__ = None
        self.actor_info = None 
        self.Camera_5mp_info = None
        self.Camera_12mp_info = None 
        self.Kinect_info = None


def img_calib(img_bgr, bgr_calib):
    rs = []
    for i in range(3):
        channel = np.array(img_bgr[:,:,i],dtype=np.double)
        X = np.stack([channel**2,channel ,np.ones_like(channel)])
        y = np.dot(bgr_calib[i].reshape(1,3),X.reshape(3,-1)).reshape(channel.shape)
        rs.append(y)
    rs_img = np.stack(rs,axis=2)
    rs_i = cv2.normalize(rs_img, None, alpha=rs_img.min(), beta=rs_img.max(), norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return rs_i


def load_cameras(annot_reader, cam_ids):
    # Load K, R, T
    cameras = {'K': [], 'R': [], 'T': [], 'D': [], 'Color_Calib': []}
    for i in range(16):
        cam_params = annot_reader.get_Calibration(cam_ids[i])
        K = cam_params['K']
        D = cam_params['D'] # k1, k2, p1, p2, k3
        c2w = cam_params['RT']
        color_calib = cam_params['Color_Calibration']
        RT = np.linalg.inv(c2w)
        cameras['K'].append(K)
        cameras['R'].append(RT[:3,:3])
        cameras['T'].append(RT[:3,3].reshape(3,))
        cameras['D'].append(D)
        cameras['Color_Calib'].append(color_calib)
    for k in cameras:
        cameras[k] = np.stack(cameras[k], axis=0)
    
    return cameras

# b = numpy.random.randn(3, 32)
# x, y, z = b
# trimesh.transformations.affine_matrix_from_points(b, numpy.stack([x, z, -y]))
BLENDER_TO_OPENGL_MATRIX = numpy.array([
    [1,  0,  0,  0],
    [0,  0,  1,  0],
    [0, -1,  0,  0],
    [0,  0,  0,  1]
], dtype=numpy.float32)
BLENDER_TO_OPENCV_MATRIX = numpy.array([
    [1,  0,  0,  0],
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [0,  0,  0,  1]
], dtype=numpy.float32)


def match_histograms(src, ref, mask):
    src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2Lab)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2Lab)
    
    l_src, a_src, b_src = cv2.split(src_lab)
    l_ref, a_ref, b_ref = cv2.split(ref_lab)

    # 匹配L通道（亮度）以解决亮度差异
    def match_channel(src_channel, ref_channel, mask):
        src_hist, _ = np.histogram(src_channel[mask > 0].flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(ref_channel[mask > 0].flatten(), 256, [0, 256])

        cdf_src = np.cumsum(src_hist).astype(np.float32)
        cdf_src /= cdf_src[-1]
        
        cdf_ref = np.cumsum(ref_hist).astype(np.float32)
        cdf_ref /= cdf_ref[-1]

        lut = np.interp(cdf_src, cdf_ref, np.arange(256))
        matched = np.interp(src_channel.flatten(), np.arange(256), lut).reshape(src_channel.shape)
        return np.clip(matched, 0, 255).astype(np.uint8)

    l_matched = match_channel(l_src, l_ref, mask)
    a_matched = match_channel(a_src, a_ref, mask)
    b_matched = match_channel(b_src, b_ref, mask)

    matched_lab = cv2.merge([l_matched, a_matched, b_matched])
    matched_rgb = cv2.cvtColor(matched_lab, cv2.COLOR_Lab2RGB)

    return matched_rgb


@DATASETS.register_module()
class NerfSynthetic(Dataset):

    def __init__(
        self, meta_files: list, world_scale: float = 1.0, rgba: bool = False, split: str = 'train', file_id: str = ''
    ) -> None:
        super().__init__()
        self.meta_files = meta_files
        self.world_scale = world_scale
        self.rgba = rgba
        self.split = split
        # self.file_id = '0031_03'
        self.file_id = file_id
        main_file = f'/fs/gamma-datasets/MannequinChallenge/dna_rendering_data/{self.file_id}.smc'
        annot_file = main_file.replace('.smc', '_annots.smc')
        self.main_reader = SMCReader(main_file)
        self.annot_reader = SMCReader(annot_file)
        self.cam_ids = [2,47,44,41,38,35,32,29,26,23,20,17,14,11,8,5]
        self.cameras = load_cameras(self.annot_reader, self.cam_ids)

    def __len__(self):
        return 1

    def load_sub(self):
        # with open(sub) as mf:
        #     meta = json.load(mf)
        # frames_i = []
        # frames_p = []
        # frames_c = []
        # frames_t = []
        # for frame in range(len(meta['frames'])):
        #     img = plotlib.imread(os.path.join(os.path.dirname(sub), meta['frames'][frame]['file_path'] + '.png'))
        #     h, w, c = img.shape
        #     x, y = w / 2, h / 2
        #     focal_length = y / numpy.tan(meta['camera_angle_x'] / 2)
        #     # scaling = 320.0 / img.shape[0]
        #     scaling = 1.0
        #     if not self.rgba:
        #         img = img[..., :3] * img[..., 3:] + (1 - img[..., 3:])
        #     # img = cv2.resize(img, [320, 320], interpolation=cv2.INTER_AREA)
        #     pose = meta['frames'][frame]['transform_matrix']
        #     frames_i.append(img)
        #     frames_p.append((numpy.array(pose) @ BLENDER_TO_OPENCV_MATRIX) * self.world_scale)
        #     frames_c.append(numpy.array([focal_length, focal_length, x, y]) * scaling)
        #     if 'time' in meta['frames'][frame]:
        #         frames_t.append(meta['frames'][frame]['time'])
        # f32 = numpy.float32
        # return dict(
        #     cond_imgs=numpy.array(frames_i, f32),
        #     cond_poses=numpy.array(frames_p, f32),
        #     cond_intrinsics=numpy.array(frames_c, f32),
        #     cond_times=numpy.array(frames_t, f32) * 2 - 1 if len(frames_t) else None
        # )
        
        Rot = np.array([
            [1, 0,  0, 0],
            [0, 0,  1, 0],
            [0, -1, 0, 0],
            [0, 0,  0, 1]
        ])
        
        
        main_file = f'/fs/gamma-datasets/MannequinChallenge/dna_rendering_data/{self.file_id}.smc'
        annot_file = main_file.replace('.smc', '_annots.smc')
        main_reader = SMCReader(main_file)
        annot_reader = SMCReader(annot_file)
        cam_ids = [2,47,44,41,38,35,32,29,26,23,20,17,14,11,8,5]
        cameras = load_cameras(annot_reader, cam_ids)
        
        
        if self.split == 'train':
            frames = [8,12,0,4]
        else:
            # frames = [1,2,3,5,6,7,9,10,11,13,14,15]
            frames = list(range(16))
        
        # get center by visual hull
        Ks = []
        bg_colors = []
        tar_w2cs = []
        tar_img = []
        tar_msks = []
        tar_msks_for_bbox = []
        
        for i, cam_idx in enumerate([8,12,0,4]):
            # Load image, mask
            image = main_reader.get_img('Camera_5mp', cam_ids[cam_idx], Image_type='color', Frame_id=0)
            color_calib = cameras['Color_Calib'][cam_idx]
            image = img_calib(image, color_calib)
            image = cv2.undistort(image, cameras['K'][cam_idx], cameras['D'][cam_idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            mask = annot_reader.get_mask(cam_ids[cam_idx], Frame_id=0)
            mask = cv2.undistort(mask, cameras['K'][cam_idx], cameras['D'][cam_idx])
            mask = mask[..., np.newaxis].astype(np.float32) / 255.0
                
            bg_color = np.ones(3).astype(np.float32)
            bg_colors.append(bg_color)
            
            image = image * mask + (bg_color*255) * (1.0 - mask)
            K = cameras['K'][cam_idx].copy()
            
            rgba = np.zeros((image.shape[0], image.shape[1], 4))
            rgba[:,:,:3] = image
            rgba[:,:,-1:] = mask.astype(np.float32)*255
            
            rgba, x1, y1, s1, s2, x2, y2 = recenter(rgba, mask.astype(np.float32)*255, bg_color*255, border_ratio=0.05)
            image = rgba[:,:,:3]
            mask = rgba[:,:,-1] / 255
            
            K[0][2] -= y1
            K[1][2] -= x1
            K[0] *= s2
            K[1] *= s1
            K[0][2] += y2
            K[1][2] += x2
            K *= (512/image.shape[0])
            K[-1,-1] = 1
            image = cv2.resize(image, (512, 512))
            mask = cv2.resize(mask, (512, 512))
            mask[mask<0.5] = 0
            mask[mask>=0.5] = 1
            
            mask_for_bbox = mask.copy()
            labeled_mask, _ = label(mask_for_bbox)
            component_sizes = np.bincount(labeled_mask.ravel())
            component_sizes[0] = 0
            largest_component_label = component_sizes.argmax()
            mask_for_bbox = (labeled_mask == largest_component_label)
            
            Ks.append(K)
            
            w2c_lara = np.eye(4)
            w2c_lara[:3,:3] = cameras['R'][cam_idx]
            w2c_lara[:3,3] = cameras['T'][cam_idx].reshape(3,)
            tar_w2cs.append(w2c_lara)
            tar_img.append(image/255)
            tar_msks.append(mask)
            tar_msks_for_bbox.append(mask_for_bbox)
        
        tar_img = np.stack(tar_img, axis=0)
        tar_img = torch.from_numpy(tar_img).clamp(0,1).float()
        tar_msks = np.stack(tar_msks, axis=0)
        tar_msks = torch.from_numpy(tar_msks).clamp(0,1).float()
        
        tar_msks_for_bbox = np.stack(tar_msks_for_bbox, axis=0)
        tar_msks_for_bbox = torch.from_numpy(tar_msks_for_bbox).clamp(0,1).float()
        tar_w2cs = np.stack(tar_w2cs, axis=0)
        tar_ixts = np.stack(Ks, axis=0)
        bg_colors = np.stack(bg_colors)
        
        hull_res = 128
        tar_msks_downsampled = F.interpolate(tar_msks_for_bbox.unsqueeze(1), size=(hull_res, hull_res), mode='bilinear', align_corners=False).squeeze(1)
        tar_msks_downsampled[tar_msks_downsampled<0.5] = 0
        tar_msks_downsampled[tar_msks_downsampled>=0.5] = 1
        tar_ixts_downsampled = tar_ixts.copy() / (512//hull_res)
        tar_ixts_downsampled[:,-1,-1] = 1
        sampled_points = visual_hull_samples(tar_msks_downsampled.detach().cpu().numpy().astype(np.float32), tar_ixts_downsampled@tar_w2cs[:,:3,:4].astype(np.float32), grid_resolution=hull_res, aabb=(-1.2, 1.2))
        center = (sampled_points.min(axis=0) + sampled_points.max(axis=0))/2
        
        
        
        
        world_scale = 0.9
        frame_id = 0
        frames_i = []
        frames_p = []
        frames_c = []
        frames_t = []
        frames_bg = []
        
        for cam_idx in frames:
            image = main_reader.get_img('Camera_5mp', cam_ids[cam_idx], Image_type='color', Frame_id=frame_id)
            color_calib = cameras['Color_Calib'][cam_idx]
            image = img_calib(image, color_calib)
            image = cv2.undistort(image, cameras['K'][cam_idx], cameras['D'][cam_idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # transform_K = np.eye(3)
            # transform_K[0][2] = image.shape[1]//2 - cameras['K'][cam_idx][0][2]
            # transform_K[1][2] = image.shape[0]//2 - cameras['K'][cam_idx][1][2]
            # image = cv2.warpPerspective(image, transform_K, (image.shape[1], image.shape[0]))
            
            mask = (annot_reader.get_mask(cam_ids[cam_idx], Frame_id=frame_id))
            mask = cv2.undistort(mask, cameras['K'][cam_idx], cameras['D'][cam_idx])
            # mask = cv2.warpPerspective(mask, transform_K, (image.shape[1], image.shape[0]))
            mask = mask.astype(np.float32) / 255.0
            
            image_bg = cv2.imread(f'/fs/gamma-projects/3dnvs_gamma/rgb-kb-tp-dna-example/extracted/{cam_ids[cam_idx]}.jpg')[:,:,::-1]
            image_bg = cv2.cvtColor(image_bg, cv2.COLOR_BGR2RGB)
            
            mask_binary = mask.copy()
            mask_binary[mask_binary<0.5] = 0
            mask_binary[mask_binary>=0.5] = 1
            mask_binary = mask_binary.astype(np.bool_)
            mask_binary = ~mask_binary
            mask_binary = mask_binary.astype(np.float32)
            image_bg = match_histograms(image_bg, image, mask_binary)
            image_bg = image_bg.astype(np.float32) / 255
            image = image.astype(np.float32) / 255
            
            if self.split == 'train':
                rgba = np.zeros((image.shape[0], image.shape[1], 4))
                rgba[...,:3] = image
                rgba[...,-1] = mask
                image = rgba
            else:
                image = image[..., :3] * mask[..., np.newaxis] + (1 - mask[..., np.newaxis])
            
            image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
            image_bg = cv2.resize(image_bg, (image_bg.shape[1]//2, image_bg.shape[0]//2))
            
            frames_i.append(image)
            frames_bg.append(image_bg)
            w2c = np.eye(4)
            w2c[:3,:3] = cameras['R'][cam_idx]
            w2c[:3,3] = cameras['T'][cam_idx]
            w2c[:3,3] += (w2c[:3,:3]@center.reshape(3,1)).reshape(3,)
            pose = np.linalg.inv(w2c)
            pose = Rot@pose
            # c2w
            frames_p.append((numpy.array(pose)) * world_scale)
            frames_c.append(numpy.array([cameras['K'][cam_idx][0][0]//2, cameras['K'][cam_idx][1][1]//2, cameras['K'][cam_idx][0][2]//2, cameras['K'][cam_idx][1][2]//2]).astype(np.float32) * 1.0)
        
        f32 = numpy.float32
        return dict(
            bg_color=bg_color,
            cond_imgs_bg=numpy.array(frames_bg, f32),
            cond_imgs=numpy.array(frames_i, f32),
            cond_poses=numpy.array(frames_p, f32),
            cond_intrinsics=numpy.array(frames_c, f32),
            cond_times=numpy.array(frames_t, f32) * 2 - 1 if len(frames_t) else None
        )

    def __getitem__(self, index):
        
        return dict(
            scene_id=DC(index, cpu_only=True),
            scene_name=DC(self.file_id, cpu_only=True),
            **self.load_sub()
        )
