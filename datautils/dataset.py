import cv2
import glob
import torch
import rasterio
import numpy as np
import einops as eo
from pathlib import Path
from torch.utils.data import Dataset
from utils.normalization import normalize
from torchvision.transforms import v2, GaussianBlur
from leafmap import leafmap
from PIL import Image

class MainDataset(Dataset):
    
    def __init__(self, path):
        self.img_dirs = sorted(glob.glob(f'{path}/*.tif'))

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        x = rasterio.open(self.img_dirs[idx]).read()
        x = normalize(x)
        x = torch.from_numpy(x)
        return self.img_dirs[idx], x.type(torch.float32)
    
class MainDatasetPNG(Dataset):
    
    def __init__(self, path):
        self.img_dirs = sorted(glob.glob(f'{path}/*.png'))

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        x = cv2.imread(self.img_dirs[idx])[:, :, ::-1]
        x = x.transpose((2, 0, 1))
        x = normalize(x)
        x = torch.from_numpy(x)
        return self.img_dirs[idx], x.type(torch.float32)

class PipelineBatchDataset(Dataset):
    def __init__(self, source, target, batchsize=16, img_size=224, zoom=19, img_file_name="pipeline/img.tif"):

        self.source = source
        self.target = target
        self.area_width = self.target[1] - self.source[1]
        self.area_height = self.target[0] - self.source[0]
        self.bbox = [source[0], source[1], target[0], target[1]]
        self.img_size = img_size
        self.batchsize = batchsize
        self.img_file_name = img_file_name
        self.batch_dim = img_size*np.sqrt(batchsize)

        print("[+] Creating dataset for area")
        try:
            pass
            # leafmap.tms_to_geotiff(output=img_file_name, bbox=self.bbox, zoom=zoom, source="Satellite")
        except Exception as e:
            print(e)
            print(self.bbox)
        
        self.img_width, self.img_height = Image.open(img_file_name).size
        self.batch_columns, self.batch_rows = self.img_width // self.batch_dim, self.img_height // self.batch_dim
        # if self.batch_columns*self.batch_dim<self.img_width: self.batch_columns += 1
        # if self.batch_rows*self.batch_dim<self.img_height: self.batch_rows += 1

        print("[+] Dataset created succesfully")

    def __len__(self):
        return self.batch_columns * self.batch_rows

    def __getitem__(self, idx):

        batch_y, batch_x = idx//self.batch_columns, idx % self.batch_columns
        batch_y, batch_x = batch_y*self.batch_dim, batch_x*self.batch_dim

        # The crop method from the Image module takes four coordinates as input.
        # The right can also be represented as (left+width)
        # and lower can be represented as (upper+height).

        # The top left coordinates correspond to (x, y) = (left, upper),
        # and the bottom right coordinates correspond to (x, y) = (right, lower).
        # The area to be cropped is left <= x <right and upper <= y <lower,
        # and the pixels of x = right andy = lower are not included.
        left, upper = batch_x, batch_y
        right, lower = min(batch_x+self.batch_dim, self.img_width), min(batch_y+self.batch_dim, self.img_height)

        img = Image.open(self.img_file_name)
        img = img.crop((left, upper, right, lower))
        img = np.array(img)

        batch_images = []
        for i in range(np.sqrt(self.batchsize)):
            for j in range(np.sqrt(self.batchsize)):
                temp = img[i*self.img_size:min((i+1)*self.img_size, self.batch_dim), j*self.img_size:min((j+1)*self.img_size, self.batch_dim)]
                batch_images.append(torch.from_numpy(temp))

class PipelineDataset(Dataset):
    def __init__(self, source, target, img_size=224, zoom=19, img_file_name="pipeline/img.tif",
                 MEAN = [0.485, 0.456, 0.406], STD=[0.229, 0.224, 0.225], normalize=True,
                 transform_images=False):

        print("[+] Creating dataset for area")
        
        self.source = source
        self.target = target
        self.area_width = self.target[1] - self.source[1]
        self.area_height = self.target[0] - self.source[0]
        self.bbox = [source[1], source[0], target[1], target[0]]
        self.img_size = img_size
        self.img_file_name = img_file_name
        self.MEAN = MEAN
        self.STD = STD
        self.normalize = normalize
        self.transform_images = transform_images

        try:
            if not Path(self.img_file_name).exists():
                print("[+] Downloding area images")
                print("File", img_file_name)
                leafmap.tms_to_geotiff(output=img_file_name, bbox=self.bbox, zoom=zoom, source="Satellite")
                print(self.bbox)
            else:
                print("[+] Area already exists, skipping download")
            self.img = Image.open(self.img_file_name)
        except Exception as e:
            print(e)
            print(self.bbox)
        
        self.img_width, self.img_height = Image.open(img_file_name).size
        self.rows, self.columns = self.img_width // self.img_size, self.img_height // self.img_size
        self.grid_width = self.area_width / (self.img_width / self.img_size)
        self.grid_height = self.area_height / (self.img_height / self.img_size)

        print("[+] Dataset created succesfully")

    def __len__(self):
        return self.columns * self.rows

    def __getitem__(self, idx):

        y, x = idx//self.columns, idx % self.columns
        y, x = y*self.img_size, x*self.img_size

        loc_y, loc_x = idx//self.columns, idx % self.columns
        loc_y, loc_x = self.target[0] - loc_y*self.grid_height, self.source[1] + loc_x*self.grid_width

        # The top left coordinates correspond to (x, y) = (left, upper),
        # and the bottom right coordinates correspond to (x, y) = (right, lower).
        left, upper, right, lower = x, y, x+self.img_size, y+self.img_size
        bbox = [loc_y, loc_x, loc_y-self.grid_height, loc_x+self.grid_width]

        img = self.img.crop((left, upper, right, lower))
        img = np.array(img)
        img = img.transpose((2, 0, 1))

        if self.transform_images: img = self._transform_images(img)
        if self.normalize: img = normalize(img, MEAN=self.MEAN, STD=self.STD)
        img = torch.from_numpy(img)

        return img.type(torch.float32), np.array(bbox)
    
    def _transform_images(self, image: np.ndarray) -> np.ndarray:
        
        jitter = v2.ColorJitter(brightness=.5, contrast=0.5, saturation=0.5, hue=.3)
        g_blur = GaussianBlur(kernel_size=3, sigma=0.5)
        tens = torch.from_numpy(image)

        # adding jitter
        if torch.rand((1, )) > 0.5:
            tens = jitter(tens)

        # adding guassian blur
        if torch.rand((1, )) > 0.5:
            tens = g_blur(tens)

        npy = tens.numpy()
        
        return npy


class BatchDataset(Dataset):
    
    def __init__(self, path):
        self.img_dirs = sorted(glob.glob(f'{path}/*.tif'))

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        
        img_name = self.img_dirs[idx].split("/")[-1]
        mask_name = f'{self.img_dirs[idx].split("/")[-1].split(".")[0]}_mask.npy'
        mask_dir = self.img_dirs[idx].replace(img_name, mask_name)
        
        x = rasterio.open(self.img_dirs[idx]).read()
        y = np.load(mask_dir)
        
        c, h, w = x.shape[0], x.shape[1], x.shape[2]
        new_h = int(h * 1.0 / 224) * 224
        new_w = int(w * 1.0 / 224) * 224
        cropped_x = x[:, :new_h, :new_w]
        cropped_y = y[:new_h, :new_w]
        
        batch_x = eo.rearrange(cropped_x, 'c (H h) (W w) -> (W H) c h w', H=new_h//224, W=new_w//224)
        batch_y = eo.rearrange(cropped_y, '(H h) (W w) -> (W H) h w', H=new_h//224, W=new_w//224)
        batch_x = normalize(batch_x)
        batch_x = torch.from_numpy(batch_x)
        batch_y = torch.from_numpy(batch_y)
        return self.img_dirs[idx], batch_x.type(torch.float32), batch_y.type(torch.int32)
    
class SpecificDataset(Dataset):
    
    def __init__(self, path):
        self.img_dirs = sorted(glob.glob(f'{path}/*.tif'))

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        x = rasterio.open(self.img_dirs[idx]).read()
        c, h, w = x.shape[0], x.shape[1], x.shape[2]
        
        center_h = 2782
        center_w = 2235
        
        x1 = x[:, center_h-112:center_h+111, center_w-112:center_w+111]
        x2 = x[:, center_h-100:center_h+123, center_w-112:center_w+111]
        x3 = x[:, center_h-100:center_h+123, center_w-100:center_w+123]
        x4 = x[:, center_h-80:center_h+143, center_w-80:center_w+143]
        x5 = x[:, center_h-173:center_h+50, center_w-80:center_w+143]
        x6 = x[:, center_h-173:center_h+50, center_w-173:center_w+50]
        x7 = x[:, center_h-200:center_h+23, center_w-200:center_w+23]
        
        x = np.stack([x1, x2, x3, x4, x5, x6, x7], axis=0)
        x = normalize(x)
        x = torch.from_numpy(x)
        return self.img_dirs[idx], x.type(torch.float32)
    
class TestDataset(Dataset):
    
    def __init__(self, path, MEAN = [0.485, 0.456, 0.406], STD=[0.229, 0.224, 0.225]):
        self.p_img_dirs = sorted(glob.glob(f'{path}/positive/*'))
        self.n_img_dirs = sorted(glob.glob(f'{path}/negative/*'))
        print(self.p_img_dirs, self.n_img_dirs)
        self.MEAN = MEAN
        self.STD = STD

    def __len__(self):
        return len(self.p_img_dirs) + len(self.n_img_dirs)

    def __getitem__(self, idx):
        if idx < len(self.n_img_dirs): # empty
            
            if self.n_img_dirs[idx].endswith('tif'): x = rasterio.open(self.n_img_dirs[idx]).read()
            else:
                x = cv2.imread(self.n_img_dirs[idx])[:, :, ::-1]
                x = x.transpose((2, 0, 1))
            
            # x = self._transform_images(x)
            x = normalize(x, MEAN=self.MEAN, STD=self.STD)
            x = torch.from_numpy(x)
            return self.n_img_dirs[idx], x.type(torch.float32), torch.tensor(0)
        else: # solar
            idx = idx % len(self.p_img_dirs)
                        
            if self.p_img_dirs[idx].endswith('tif'): x = rasterio.open(self.p_img_dirs[idx]).read()
            else: 
                x = cv2.imread(self.p_img_dirs[idx])[:, :, ::-1]
                x = x.transpose((2, 0, 1))
            
            # x = self._transform_images(x)
            x = normalize(x, MEAN=self.MEAN, STD=self.STD)
            x = torch.from_numpy(x)
            return self.p_img_dirs[idx], x.type(torch.float32), torch.tensor(1.0)
    
    def _transform_images(self, image: np.ndarray) -> np.ndarray:
        
        jitter = v2.ColorJitter(brightness=.5, contrast=0.5, saturation=0.5, hue=.3)
        g_blur = GaussianBlur(kernel_size=3, sigma=0.5)
        tens = torch.from_numpy(image)

        # adding jitter
        if torch.rand((1, )) > 0.5:
            tens = jitter(tens)

        # adding guassian blur
        if torch.rand((1, )) > 0.5:
            tens = g_blur(tens)

        npy = tens.numpy()
        
        return npy
