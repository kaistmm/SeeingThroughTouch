import os
import json
from typing import Optional, Callable
from torch.utils.data import Dataset
from STT import ModalityType
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image 
import torchvision.transforms.functional as TF
import numpy as np
import random
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
TOUCH_AND_GO_METADATA_DIR = PROJECT_ROOT / "datasets" / "touch_and_go" / "metadata"
WEBMATERIAL_METADATA_DIR = PROJECT_ROOT / "datasets" / "WebMaterial" / "metadata"

# Dataset Statistics
RGB_MEAN = np.array([0.485, 0.456, 0.406])
RGB_STD = np.array([0.229, 0.224, 0.225])

TAC_MEAN = np.array([0.54390774, 0.51392555, 0.54791247])
TAC_STD = np.array([0.1421082,  0.11569928, 0.13259748])
    
def bring_specific_tactile_frame(target_frame, tactile_frame, touch_instance_json):
    """
    Given a tactile frame path, look up the touch instance JSON to find the frame at
    `target_frame` position ('start', 'middle', or 'end') within the same touch instance.
    """
    dir_path = os.path.dirname(tactile_frame)
    video_id = dir_path.split("/")[-2]
    file_name = os.path.basename(tactile_frame)
    frame_idx = str(int(file_name.split(".")[0]))
    mid_idx = touch_instance_json[video_id][frame_idx][target_frame]
    mid_frame_path = os.path.join(dir_path, f"{mid_idx:010d}.jpg")
    return mid_frame_path

def to_pil(img : torch.Tensor):
    img = np.moveaxis(img.numpy()*255, 0, -1)
    return Image.fromarray(img.astype(np.uint8))

def unnormalize_fn(mean : tuple, std : tuple) -> transforms.Compose:
    """
    returns a transformation that turns torch tensor to PIL Image
    """
    return transforms.Compose(
        [
            transforms.Normalize(
                mean=tuple(-m / s for m, s in zip(mean, std)),
                std=tuple(1.0 / s for s in std),
            ),
            transforms.Lambda(lambda x: torch.clamp(x, 0., 1.)), 
            transforms.ToPILImage(),
        ]
    )

#---------------- Augmentation for 224*224 resized images -------------------
class SimulateRatioDistortionAndRotation(nn.Module):
    """
    Simulates the distortion that occurs when a high-res image is augmented
    (aspect ratio change + rotation) and then resized to a square.

    Args:
        original_size (tuple): (width, height) of the original image, e.g. (640, 480).
        p (float): Probability of applying this transform (0.0 ~ 1.0).
        padding (bool): If True, reproduces padding artifacts from rotation.
                        If False, only reproduces content distortion.
    """
    def __init__(self, original_size, p=0.5, padding=True):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be between 0.0 and 1.0, got {p}.")

        self.p = p
        self.padding = padding
        w, h = original_size

        # After 90° rotation, original (W, H) becomes (H, W).
        # Store the aspect ratio (W/H) of the rotated image.
        self.rotated_aspect_ratio_wh = h / w

    def forward(self, img):
        # Skip transform with probability (1 - p)
        if random.random() > self.p:
            return img

        # Assume square input; use height as reference size
        size = transforms.functional.get_image_size(img)[0]

        # Compute width for target aspect ratio (e.g. 224 * 0.75 = 168)
        new_width = int(size * self.rotated_aspect_ratio_wh)

        # 1. Squeeze to target aspect ratio
        img = transforms.functional.resize(img, (new_width, size))

        # 2. Rotate 90° to match original image orientation
        img = transforms.functional.rotate(img, 90)

        # 3. If not reproducing padding, restore to square to isolate distortion only
        if not self.padding:
            img = transforms.functional.resize(img, (size, size))
            
        return img
#-------------------------------------------------------------------
class RandomDiscreteRotation(nn.Module):
    """Rotate by one of the given angles."""
    def __init__(self, angles):
        self.angles = angles
    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

TO_TENSOR = transforms.Compose([
    transforms.ToTensor()
])

# Image Augmentations
RGB_AUGMENTS = transforms.Compose([
    transforms.RandomResizedCrop(
        (224, 224), 
        scale=(0.6, 1.0),  
        ratio=(0.75, 1.33), 
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.RandomHorizontalFlip(),
    RandomDiscreteRotation([0, 90]),
    transforms.RandomApply([transforms.ColorJitter(
        brightness=(0.6, 1.4),
        contrast=(0.6, 1.4),
        saturation=0.4,
        hue=0.1
    )], p=.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(9, sigma=(.5, 1))], p=.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
])

RGB_PREPROCESS = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=RGB_MEAN,
        std=RGB_STD,
    ),
])

# Tactile Augmentations
TAC_AUGMENTS = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.ColorJitter(
        brightness=(0.9, 1.1),
        contrast=(.9, 1.1),
        saturation=0.2,
        hue=0.05
    )], p=.8),
    SimulateRatioDistortionAndRotation(original_size=(640,480), p=0.5, padding=False),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=TAC_MEAN,
        std=TAC_STD,
    ),
])

TAC_PREPROCESS = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=TAC_MEAN,
        std=TAC_STD,
    ),
])

#------------------ Load Modality Data -------------------

# Vision
def load_vision_data(
    path : str,
    transform_rgb = transforms.Compose([
        transforms.Resize(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=RGB_MEAN,
            std=RGB_STD,
        ),
    ]),
    device : str = None,
):
    """Load a vision image from `path`, apply transform, and optionally move to device."""
    rgb = Image.open(path)
    if transform_rgb is not None: 
        rgb = transform_rgb(rgb)
    if device is not None:
        rgb = rgb.to(device)
    return rgb

# Tactile
def load_tactile_data(
    path : str,
    transform_tac = transforms.Compose([
        transforms.Resize(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=TAC_MEAN,
            std=TAC_STD,
        ),
    ]),
    device : str = None,
    ):
    """Load a tactile image from `path`, apply transform, and optionally move to device."""
    tac = Image.open(path)
    if transform_tac is not None:
        tac = transform_tac(tac)
    if device is not None:
        tac = tac.to(device)
    return tac

class TouchAndGo_WebMaterial_MDP(Dataset):
    """
    Touch-instance based dataset with support for Material Diversity-based(MDP) pairing.
    
    MDP 
    1. 'In-domain': Pairs vision from original touch instances with tactile from different touch instances of same category
       - Epoch length: 2N (N original + N in-domain MDP)
    2. 'Out-domain': Pairs external vision images with tactile from same-category touch instances
       - Epoch length: N + M (N original + M out-domain MDP)
    
    Key features:
    - Touch-instance based dynamic sampling for original data
    - Material Diversity-based pairing samples to enhance intra-category tactile learning
    - MDP: extend vision from In-domain/Out-domain sources (TouchAndGo or Web-Material) while keeping tactile grounded in TouchAndGo
    - Reproducible random sampling per epoch
    """
    
    def __init__(self, root_dir: str, transform_rgb: Optional[Callable] = None, transform_tac: Optional[Callable] = None,
                 split: str = 'train', random_seed: int = 42, 
                 modality_types = [ModalityType.VISION, ModalityType.TACTILE],
                 device: str = 'cpu', rgb_size=[224, 224], tac_size=[224, 224], im_scale_range=[.12, .18], 
                 randomize_crop=False, test_split_type="no_inter",
                 TouchInstance_file=None, eval_mode="retrieval",
                 MDP_mode=None, WebMaterial_file=None, WebMaterial_base_dir=None,
                 curriculum_epoch=None):
        """
        Args:
            TouchInstance_file (str): Path to touch instance file for training (format: "video_id,start,end,category")
            eval_mode (str): Evaluation mode - "retrieval" or "semseg"
            MDP_mode (str): MDP mode - None, 'In-domain', or 'Out-domain'
            WebMaterial_file (str): Path to Web-Material file
            WebMaterial_base_dir (str): Base directory for Web-Material images
            curriculum_epoch (int): Epoch to switch from Case 1 only to MDP mode. None to disable curriculum learning.
        """
        
        self.rgb_size = rgb_size
        self.tac_size = tac_size
        self.im_scale_range = im_scale_range
        self.randomize_crop = randomize_crop
        self.split = split
        self.test_split_type = test_split_type
        self.random_seed = random_seed
        self.eval_mode = eval_mode
        self.current_epoch = 0
        self.MDP_mode = MDP_mode if split == 'train' else None  # Only apply MDP to training
        self.curriculum_epoch = curriculum_epoch
        
        print("root_dir: ", root_dir)
        self.dataset_dir = os.path.join(root_dir, "dataset_224")
        print("dataset_dir: ", self.dataset_dir)
        
        assert split in ["train", "test", "val"]
        assert test_split_type in ["original", "no_inter"]
        assert self.MDP_mode in [None, 'In-domain', 'Out-domain']
        print("split: ", split)
        print(f"test_split_type: {test_split_type}")
        print(f"MDP_mode: {self.MDP_mode}")
        print(f"curriculum_epoch: {self.curriculum_epoch}")

        if not isinstance(root_dir, list):
            self.root_dir = [root_dir]
        else:
            self.root_dir = root_dir
            
        self.transform_rgb = transform_rgb
        self.transform_tac = transform_tac
        self.device = device
        self.modality_types = modality_types
        
        if self.eval_mode == "semseg":
            test_nointer_no_others_json_path = str(TOUCH_AND_GO_METADATA_DIR / "test_nointer_touch_instances.json")
            with open(test_nointer_no_others_json_path, "r") as f:
                self.test_nointer_touch_instances = json.load(f)
        
        # touch_instance mode
        if TouchInstance_file is not None and split == "train":
            print(f"[INFO: DATASET_MDP] Loading touch instances from {TouchInstance_file}")
            self.touch_instance_lines = []
            with open(TouchInstance_file, 'r') as f:
                self.touch_instance_lines = [line.strip() for line in f if line.strip()]
            
            random.Random(self.random_seed).shuffle(self.touch_instance_lines)
            print(f"Loaded {len(self.touch_instance_lines)} touch instances (shuffled)")
            
            self.category_to_touch_instances = {}
            for i, line in enumerate(self.touch_instance_lines):
                parts = line.split(',')
                category = int(parts[3])
                if category not in self.category_to_touch_instances:
                    self.category_to_touch_instances[category] = []
                self.category_to_touch_instances[category].append(i)
            
            print(f"Built category mapping: {[(cat, len(instances)) for cat, instances in sorted(self.category_to_touch_instances.items())]}")
            
            # Load MDP file if in Out-domain mode
            self.MDP_out_lines = []
            self.WebMaterial_base_dir = WebMaterial_base_dir or "/path/to/SeeingThroughTouch/datasets/WebMaterial/train/image" # Update this path to your local Web-Material images directory
            
            if self.MDP_mode == 'Out-domain' and WebMaterial_file is not None:
                print(f"[INFO: DATASET_MDP] Loading Out-domain MDP from {WebMaterial_file}")
                with open(WebMaterial_file, 'r') as f:
                    self.MDP_out_lines = [line.strip() for line in f if line.strip()]
                print(f"Loaded {len(self.MDP_out_lines)} Out-domain MDP samples")
            self.paths = None  
            
        else:
            # Frame-based loading for test/val
            print(f"[INFO: DATASET_MDP] Using frame-based loading (not touch instance mode)")
            self.touch_instance_lines = None
            self.category_to_touch_instances = {}
            self.MDP_out_lines = []
            self.data = {"tactile": [], "vision": []}
            
            for data_dir in self.root_dir:
                metadata_dir = os.path.join(data_dir, "metadata")
                if split == "val":
                    if test_split_type == "no_inter":
                        split_txt = os.path.join(metadata_dir, "test_1118_touch_instances.txt")
                    else:
                        split_txt = os.path.join(metadata_dir, "test_1113.txt")
                else:  # test split
                    print(f"[INFO] eval mode: {self.eval_mode}")
                    if test_split_type == "no_inter": # Leakage-free split
                        if self.eval_mode == "retrieval":
                            split_txt = os.path.join(metadata_dir, "test_1118_touch_instances.txt")
                        elif self.eval_mode == "semseg":
                            split_txt = os.path.join(metadata_dir, "test_579_semseg.txt")
                        else:
                            split_txt = os.path.join(metadata_dir, "test_1118_touch_instances.txt")
                    else: # Original split
                        split_txt = os.path.join(metadata_dir, "test_1113.txt")

                if os.path.exists(split_txt):
                    print(f"Loading text file: {split_txt}.")
                    with open(split_txt, "r") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            test_sample, test_category = line.strip().split(",")
                            sample_id, frame_id = test_sample.split("/")
                            if ("20220318_020426" in sample_id) or ("20220318_021048" in sample_id):
                                continue

                            tactile_sample = os.path.join(self.dataset_dir, sample_id, "gelsight_frame", frame_id)
                            if self.eval_mode == "semseg":
                                tactile_sample = bring_specific_tactile_frame("middle", tactile_sample, self.test_nointer_touch_instances)
                            
                            vision_sample = os.path.join(self.dataset_dir, sample_id, "video_frame", frame_id)
                            self.data["tactile"].append(tactile_sample)
                            self.data["vision"].append(vision_sample)
                else:
                    print(f"Split file {split_txt} not found in {data_dir}.")
            
            self.paths = self.data
            print(f"Loaded {len(self.paths['vision'])} samples for {split}")

    def __repr__(self):
        return f"{self.__class__.__name__}(root_dir={self.root_dir}, split={self.split}, MDP_mode={self.MDP_mode}, touch_instance_mode={self.touch_instance_lines is not None})"

    def __len__(self):
        if self.touch_instance_lines is not None:
            base_len = len(self.touch_instance_lines)
            
            # Curriculum learning: only use Case 1 before curriculum_epoch
            if self.curriculum_epoch is not None and self.current_epoch < self.curriculum_epoch:
                return base_len
            
            # After curriculum_epoch, use MDP_mode
            if self.MDP_mode == 'In-domain':
                return 2 * base_len  # N original + N MDP in-domain
            elif self.MDP_mode == 'Out-domain':
                return base_len + len(self.MDP_out_lines)  # N original + M MDP out-domain
            return base_len
        else:
            return len(self.paths["vision"])

    def set_epoch(self, epoch):
        """Set current epoch for reproducible touch instance sampling"""
        self.current_epoch = epoch
        if self.touch_instance_lines is not None:
            N = len(self.touch_instance_lines)
            
            # Curriculum learning status check
            if self.curriculum_epoch is not None:
                if epoch < self.curriculum_epoch:
                    print(f"[CURRICULUM] Epoch {epoch}: Using only Case 1 (original touch instances) - {epoch}/{self.curriculum_epoch}")
                elif epoch == self.curriculum_epoch:
                    print(f"[CURRICULUM] Epoch {epoch}: Switching to MDP_mode '{self.MDP_mode}'")

            # Regular logging
            if self.curriculum_epoch is None or epoch >= self.curriculum_epoch:
                if self.MDP_mode == 'In-domain':
                    print(f"[DATASET] Epoch {epoch}: {N} original touch instances + {N} In-domain MDP = {2*N} total samples")
                elif self.MDP_mode == 'Out-domain':
                    M = len(self.MDP_out_lines)
                    print(f"[DATASET] Epoch {epoch}: {N} original touch instances + {M} Out-domain MDP = {N+M} total samples")
                else:
                    print(f"[DATASET] Epoch {epoch}: {N} touch instances (no MDP)")
            else:
                print(f"[DATASET] Epoch {epoch}: {N} touch instances (curriculum phase)")
            
            print(f"[DATASET] Epoch {epoch}: Reproducible sampling with seed {self.random_seed + epoch}")

    def load_vision_data(self, path: str):
        return load_vision_data(path, transform_rgb=self.transform_rgb)

    def load_tactile_data(self, path: str):
        return load_tactile_data(path, transform_tac=self.transform_tac)

    def __getitem__(self, index):
        item = OrderedDict()
        
        if self.touch_instance_lines is not None:
            N = len(self.touch_instance_lines)
            
            # Reproducible random sampling: same epoch + index = same frames
            epoch_seed = self.random_seed + self.current_epoch * 100000 + index
            rng = random.Random(epoch_seed)
            
            # Check if in curriculum learning phase
            in_curriculum_phase = (self.curriculum_epoch is not None and 
                                   self.current_epoch < self.curriculum_epoch)
            
            # Case 1: Original touch_instance sampling (always executed when index < N or in curriculum phase)
            if index < N:
                parts = self.touch_instance_lines[index].split(',')
                video_id, start, end, category = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
                
                vision_frame = rng.randint(start, end)
                tactile_frame = rng.randint(start, end)
                
                vision_path = os.path.join(
                    self.dataset_dir, video_id, "video_frame", f"{vision_frame:010d}.jpg"
                )
                tactile_path = os.path.join(
                    self.dataset_dir, video_id, "gelsight_frame", f"{tactile_frame:010d}.jpg"
                )
            
            # Case 2 & 3: Only execute if NOT in curriculum phase
            elif not in_curriculum_phase:
                # Case 2: In-domain MDP (index >= N and MDP_mode == 'In-domain')
                if self.MDP_mode == 'In-domain' and index < 2 * N:
                    original_idx = index - N
                    parts = self.touch_instance_lines[original_idx].split(',')
                    video_id, start, end, category = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
                    
                    # Vision: from original touch instance
                    vision_frame = rng.randint(start, end)
                    vision_path = os.path.join(
                        self.dataset_dir, video_id, "video_frame", f"{vision_frame:010d}.jpg"
                    )
                    
                    # Tactile: from different touch instance of same category
                    same_category_instance_indices = [i for i in self.category_to_touch_instances[category] if i != original_idx]
                    
                    if same_category_instance_indices:
                        other_instance_idx = rng.choice(same_category_instance_indices)
                        other_parts = self.touch_instance_lines[other_instance_idx].split(',')
                        other_video_id, other_start, other_end = other_parts[0], int(other_parts[1]), int(other_parts[2])
                        
                        tactile_frame = rng.randint(other_start, other_end)
                        tactile_path = os.path.join(
                            self.dataset_dir, other_video_id, "gelsight_frame", f"{tactile_frame:010d}.jpg"
                        )
                    else:
                        # Fallback: use original touch instance if no other touch instances available
                        tactile_frame = rng.randint(start, end)
                        tactile_path = os.path.join(
                            self.dataset_dir, video_id, "gelsight_frame", f"{tactile_frame:010d}.jpg"
                        )
                
                # Case 3: MDP Out-domain (index >= N and MDP_mode == 'Out-domain')
                elif self.MDP_mode == 'Out-domain' and index < N + len(self.MDP_out_lines):
                    extension_idx = index - N
                    extension_line = self.MDP_out_lines[extension_idx]
                    vision_path_rel, category = extension_line.split(',')
                    category = int(category)
                    
                    # Category mapping for Web-Material dataset
                    category_to_WebMaterial = {
                        "0": "Concrete",
                        "1": "Plastic", 
                        "2": "Glass",
                        "3": "Wood",
                        "4": "Metal",
                        "5": "Brick",
                        "6": "Tile",
                        "7": "Leather",
                        "8": "Fabric",
                        "10": "Rubber",
                        "11": "Paper",
                        "12": "Tree",
                        "13": "Grass",
                        "14": "Soil",
                        "15": "Rock",
                        "16": "Gravel",
                        "17": "Sand",
                        "18": "Plants"
                    }
                    
                    # Vision: from Web-Material with category directory
                    category_dir = category_to_WebMaterial.get(str(category), "")
                    if category_dir:
                        if vision_path_rel.startswith(f"{category_dir}/"):
                            vision_path = os.path.join(self.WebMaterial_base_dir, vision_path_rel)
                        else:
                            vision_path = os.path.join(self.WebMaterial_base_dir, category_dir, vision_path_rel)
                    else:
                        print(f"[WARNING] Category {category} not found in category mapping, using path without category dir")
                        vision_path = os.path.join(self.WebMaterial_base_dir, vision_path_rel)
                    
                    # Tactile: random touch instance of same category -> random frame
                    if category in self.category_to_touch_instances and self.category_to_touch_instances[category]:
                        selected_instance_idx = rng.choice(self.category_to_touch_instances[category])
                        instance_parts = self.touch_instance_lines[selected_instance_idx].split(',')
                        instance_video_id, instance_start, instance_end = instance_parts[0], int(instance_parts[1]), int(instance_parts[2])
                        
                        tactile_frame = rng.randint(instance_start, instance_end)
                        tactile_path = os.path.join(
                            self.dataset_dir, instance_video_id, "gelsight_frame", f"{tactile_frame:010d}.jpg"
                        )
                    else:
                        # Fallback: use first instance if category not found
                        print(f"[WARNING] Category {category} not found in touch instances, using fallback")
                        fallback_parts = self.touch_instance_lines[0].split(',')
                        fallback_video_id, fallback_start, fallback_end = fallback_parts[0], int(fallback_parts[1]), int(fallback_parts[2])
                        
                        tactile_frame = rng.randint(fallback_start, fallback_end)
                        tactile_path = os.path.join(
                            self.dataset_dir, fallback_video_id, "gelsight_frame", f"{tactile_frame:010d}.jpg"
                        )
                
                else:
                    raise IndexError(f"Index {index} out of range for dataset length {len(self)}")
            
            else:
                # In curriculum phase, index should always be < N
                raise IndexError(f"Index {index} out of range during curriculum phase (max: {N-1})")
        
        else:
            # Frame-based mode: Use pre-loaded paths
            vision_path = self.paths["vision"][index]
            tactile_path = self.paths["tactile"][index]
        
        # Load data using existing helper functions
        if ModalityType.VISION in self.modality_types:
            images = self.load_vision_data(vision_path)
            item[ModalityType.VISION] = [images]
        if ModalityType.TACTILE in self.modality_types:
            tactiles = self.load_tactile_data(tactile_path)
            item[ModalityType.TACTILE] = [tactiles]
        
        return item
    
class TouchAndGoDataset_TouchInstance(Dataset):
    """
    Touch-instance based dataset with dynamic frame sampling.

    Uses touch instance files (video_id, start_frame, end_frame, category) for dynamic
    frame sampling within tactile event segments. Each touch instance represents a
    continuous tactile event where vision and tactile frames are randomly sampled
    from the same temporal range.

    - Train split: Uses touch instance file for dynamic frame pair sampling (no fixed pairs)
    - Test/Val split: Uses traditional frame-based loading for evaluation consistency
    - Supports eval_mode for different evaluation protocols (retrieval, semseg)
    """

    def __init__(self, root_dir: str, transform_rgb: Optional[Callable] = None, transform_tac: Optional[Callable] = None,
                 split: str = 'train', random_seed: int = 42,
                 modality_types = [ModalityType.VISION, ModalityType.TACTILE],
                 device: str = 'cpu', rgb_size=[224, 224], tac_size=[224, 224], im_scale_range=[.12, .18],
                 randomize_crop=False, test_split_type="no_inter",
                 TouchInstance_file=None, eval_mode="retrieval"):
        """
        Args:
            TouchInstance_file (str): Path to touch instance file for training (format: "video_id,start,end,category")
            eval_mode (str): Evaluation mode - "retrieval" or "semseg"
        """

        self.rgb_size = rgb_size
        self.tac_size = tac_size
        self.im_scale_range = im_scale_range
        self.randomize_crop = randomize_crop
        self.split = split
        self.test_split_type = test_split_type
        self.random_seed = random_seed
        self.eval_mode = eval_mode
        self.current_epoch = 0

        print("root_dir: ", root_dir)
        self.dataset_dir = os.path.join(root_dir, "dataset_224")
        print("dataset_dir: ", self.dataset_dir)

        assert split in ["train", "test", "val"]
        assert test_split_type in ["original", "no_inter"]
        print("split: ", split)
        print(f"test_split_type: {test_split_type}")

        if not isinstance(root_dir, list):
            self.root_dir = [root_dir]
        else:
            self.root_dir = root_dir

        self.transform_rgb = transform_rgb
        self.transform_tac = transform_tac
        self.device = device
        self.modality_types = modality_types

        if self.eval_mode == "semseg":
            test_nointer_no_others_json_path = str(TOUCH_AND_GO_METADATA_DIR / "test_nointer_touch_instances.json")
            with open(test_nointer_no_others_json_path, "r") as f:
                self.test_nointer_touch_instances = json.load(f)

        # Touch instance mode for training
        if TouchInstance_file is not None and split == "train":
            print(f"[INFO: DATASET] Loading touch instances from {TouchInstance_file}")
            self.touch_instance_lines = []
            with open(TouchInstance_file, 'r') as f:
                self.touch_instance_lines = [line.strip() for line in f if line.strip()]

            random.Random(self.random_seed).shuffle(self.touch_instance_lines)
            print(f"Loaded {len(self.touch_instance_lines)} touch instances (shuffled)")
            self.paths = None

        else:
            # Frame-based loading for test/val
            print(f"[INFO: DATASET] Using frame-based loading (not touch instance mode)")
            self.touch_instance_lines = None
            self.data = {"tactile": [], "vision": []}

            for data_dir in self.root_dir:
                metadata_dir = os.path.join(data_dir, "metadata")
                if split == "val":
                    if test_split_type == "no_inter":
                        split_txt = os.path.join(metadata_dir, "test_1118_touch_instances.txt")
                    else:
                        split_txt = os.path.join(metadata_dir, "test_1113.txt")
                else:  # test split
                    print(f"[INFO] eval mode: {self.eval_mode}")
                    if test_split_type == "no_inter":
                        if self.eval_mode == "retrieval":
                            split_txt = os.path.join(metadata_dir, "test_1118_touch_instances.txt")
                        elif self.eval_mode == "semseg":
                            split_txt = os.path.join(metadata_dir, "test_579_semseg.txt")
                        else:
                            split_txt = os.path.join(metadata_dir, "test_1118_touch_instances.txt")
                    else:
                        split_txt = os.path.join(metadata_dir, "test_1113.txt")

                if os.path.exists(split_txt):
                    print(f"Loading text file: {split_txt}.")
                    with open(split_txt, "r") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            test_sample, test_category = line.strip().split(",")
                            sample_id, frame_id = test_sample.split("/")
                            if ("20220318_020426" in sample_id) or ("20220318_021048" in sample_id):
                                continue

                            tactile_sample = os.path.join(self.dataset_dir, sample_id, "gelsight_frame", frame_id)
                            if self.eval_mode == "semseg":
                                tactile_sample = bring_specific_tactile_frame("middle", tactile_sample, self.test_nointer_touch_instances)

                            vision_sample = os.path.join(self.dataset_dir, sample_id, "video_frame", frame_id)
                            self.data["tactile"].append(tactile_sample)
                            self.data["vision"].append(vision_sample)
                else:
                    print(f"Split file {split_txt} not found in {data_dir}.")

            self.paths = self.data
            print(f"Loaded {len(self.paths['vision'])} samples for {split}")

    def __repr__(self):
        return f"{self.__class__.__name__}(root_dir={self.root_dir}, split={self.split}, touch_instance_mode={self.touch_instance_lines is not None})"

    def __len__(self):
        if self.touch_instance_lines is not None:
            return len(self.touch_instance_lines)
        else:
            return len(self.paths["vision"])

    def set_epoch(self, epoch):
        """Set current epoch for reproducible touch instance sampling"""
        self.current_epoch = epoch
        if self.touch_instance_lines is not None:
            print(f"[DATASET] Epoch {epoch}: {len(self.touch_instance_lines)} touch instances (seed {self.random_seed + epoch})")

    def load_vision_data(self, path: str):
        return load_vision_data(path, transform_rgb=self.transform_rgb)

    def load_tactile_data(self, path: str):
        return load_tactile_data(path, transform_tac=self.transform_tac)

    def __getitem__(self, index):
        item = OrderedDict()

        if self.touch_instance_lines is not None:
            # Touch instance mode: Dynamic path generation with reproducible sampling
            parts = self.touch_instance_lines[index].split(',')
            video_id, start, end, category = parts[0], int(parts[1]), int(parts[2]), int(parts[3])

            # Reproducible random sampling: same epoch + index = same frames
            epoch_seed = self.random_seed + self.current_epoch * 100000 + index
            rng = random.Random(epoch_seed)

            vision_frame = rng.randint(start, end)
            tactile_frame = rng.randint(start, end)

            vision_path = os.path.join(
                self.dataset_dir, video_id, "video_frame", f"{vision_frame:010d}.jpg"
            )
            tactile_path = os.path.join(
                self.dataset_dir, video_id, "gelsight_frame", f"{tactile_frame:010d}.jpg"
            )
        else:
            # Frame-based mode: Use pre-loaded paths
            vision_path = self.paths["vision"][index]
            tactile_path = self.paths["tactile"][index]

        if ModalityType.VISION in self.modality_types:
            images = self.load_vision_data(vision_path)
            item[ModalityType.VISION] = [images]
        if ModalityType.TACTILE in self.modality_types:
            tactiles = self.load_tactile_data(tactile_path)
            item[ModalityType.TACTILE] = [tactiles]

        return item


class TouchAndGoDataset(Dataset):
    """
    Frame-based Touch-and-Go dataset.

    Loads (vision, tactile) pairs from a fixed split text file. Supports
    `eval_mode="semseg"` which remaps each test tactile frame to the middle
    frame of its touch instance for semantic-segmentation evaluation.
    """

    def __init__(self, root_dir: str, transform_rgb: Optional[Callable] = None, transform_tac: Optional[Callable] = None,
                 split: str = 'train', random_seed: int = 42,
                 modality_types = [ModalityType.VISION, ModalityType.TACTILE],
                 device: str = 'cpu', rgb_size=[224, 224], tac_size=[224, 224], im_scale_range=[.12, .18],
                 randomize_crop=False, test_split_type="no_inter", eval_mode="retrieval"):
        self.rgb_size = rgb_size
        self.tac_size = tac_size
        self.im_scale_range = im_scale_range
        self.randomize_crop = randomize_crop
        self.split = split
        self.test_split_type = test_split_type
        self.random_seed = random_seed
        self.eval_mode = eval_mode

        print("root_dir: ", root_dir)
        self.dataset_dir = os.path.join(root_dir, "dataset_224")
        print("dataset_dir: ", self.dataset_dir)

        assert split in ["train", "test", "val"]
        assert test_split_type in ["original", "no_inter"]
        print("split: ", split)
        print(f"test_split_type: {test_split_type}")

        if not isinstance(root_dir, list):
            self.root_dir = [root_dir]
        else:
            self.root_dir = root_dir

        self.transform_rgb = transform_rgb
        self.transform_tac = transform_tac
        self.device = device
        self.modality_types = modality_types

        if self.eval_mode == "semseg":
            test_nointer_no_others_json_path = str(TOUCH_AND_GO_METADATA_DIR / "test_nointer_touch_instances.json")
            with open(test_nointer_no_others_json_path, "r") as f:
                self.test_nointer_touch_instances = json.load(f)

        print(f"[INFO: DATASET] Using {test_split_type.upper()} version train and test split")
        self.data = {"tactile": [], "vision": []}

        for data_dir in self.root_dir:
            split_txt = self._resolve_split_file(data_dir)
            if not os.path.exists(split_txt):
                print(f"Split file {split_txt} not found in {data_dir}.")
                continue

            print(f"Loading text file: {split_txt}.")
            with open(split_txt, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    test_sample, _test_category = line.strip().split(",")
                    sample_id, frame_id = test_sample.split("/")
                    if ("20220318_020426" in sample_id) or ("20220318_021048" in sample_id):
                        continue

                    tactile_sample = os.path.join(self.dataset_dir, sample_id, "gelsight_frame", frame_id)
                    if self.eval_mode == "semseg":
                        tactile_sample = bring_specific_tactile_frame(
                            "middle", tactile_sample, self.test_nointer_touch_instances
                        )
                    vision_sample = os.path.join(self.dataset_dir, sample_id, "video_frame", frame_id)
                    self.data["tactile"].append(tactile_sample)
                    self.data["vision"].append(vision_sample)

        self.paths = self.data
        if split == "train":
            print("number of training samples: ", len(self.paths["vision"]))
        elif split == "val":
            print("number of validation(=test) samples: ", len(self.paths["vision"]))
        else:
            print("number of testing samples: ", len(self.paths["vision"]))

    def _resolve_split_file(self, data_dir: str) -> str:
        """Return the split text file path for the current split/eval_mode."""
        metadata_dir = os.path.join(data_dir, "metadata")
        if self.split == "train":
            if self.test_split_type == "no_inter":
                return os.path.join(metadata_dir, "train_nointer_touch_instances.txt")
            return os.path.join(metadata_dir, "train.txt")

        if self.split == "val":
            if self.test_split_type == "no_inter":
                return os.path.join(metadata_dir, "test_1118_touch_instances.txt")
            return os.path.join(metadata_dir, "test_1113.txt")

        # test split
        if self.test_split_type == "no_inter":
            return os.path.join(metadata_dir, "test_579_semseg.txt")
        return os.path.join(metadata_dir, "test_1113.txt")

    def __repr__(self):
        return (f"{self.__class__.__name__}(root_dir={self.root_dir}, split={self.split}, "
                f"modality_types={self.modality_types}, test_split_type={self.test_split_type}, "
                f"eval_mode={self.eval_mode})")

    def __len__(self):
        return len(self.paths["vision"])

    def load_vision_data(self, path: str):
        return load_vision_data(path, transform_rgb=self.transform_rgb)

    def load_tactile_data(self, path: str):
        return load_tactile_data(path, transform_tac=self.transform_tac)

    def __getitem__(self, index):
        item = OrderedDict()
        if ModalityType.VISION in self.modality_types:
            item[ModalityType.VISION] = [self.load_vision_data(self.paths["vision"][index])]
        if ModalityType.TACTILE in self.modality_types:
            item[ModalityType.TACTILE] = [self.load_tactile_data(self.paths["tactile"][index])]
        return item
