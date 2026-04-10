# Dataset Preparation
## 1. Download

Please follow the instructions in the links below to download the training and evaluation datasets and corresponding segmentation masks.

<table border="1">
  <tr>
    <td><b>Dataset</b></td>
    <td><b>Usage</b></td>
    <td><b>Image</b></td>
  </tr>
  <tr>
    <td>Touch and Go</td>
    <td>Train, Eval</td>
    <td><a href="https://touch-and-go.github.io/">here</a></td>
  </tr>
  <tr>
    <td>WebMaterial</td>
    <td>Train, Eval</td>
    <td>Section 1.1</td>
  </tr>
  <tr>
    <td>OpenSurfaces</td>
    <td>Eval</td>
    <td><a href="http://opensurfaces.cs.cornell.edu/publications/opensurfaces/">here</a></td>
  </tr>
</table>

### 1.1 WebMaterial Dataset

1. **Download base datasets**
   - [MINC-2500](http://opensurfaces.cs.cornell.edu/publications/minc/): Follow the official download instructions.
   - [ILSVRC2012](https://www.image-net.org/challenges/LSVRC/2012/browse-synsets.php): Only the `n09421951` (sandbar) class is required.

2. **Reorganize image paths**  
   Using `{split}_minc2500_path.txt` and `{split}_sand_imagenet.txt` in `datasets/WebMaterial/url/{split}/`,
   move each image from its original path (right column) to the designated path (left column).

3. **Download web-crawled images**  
   Using `{split}_Web_url.txt` in `datasets/WebMaterial/url/{split}/`,
   download each image from the URL (right column) to the designated path (left column).

### 1.2 Evaluation Masks
Download the mask zip files from [here](https://huggingface.co/seongyu/SeeingThroughTouch/tree/main/eval_masks). Extract each file into its corresponding *datasets/{dataset}/\*/mask* directory as shown below.

## 2. Data Structure
The structure of datasets directory is as follows:

```
datasets/
│
├── touch_and_go/
│   ├── dataset_224/              # Visuo-tactile RGB frames organized by video IDs (e.g., 20220607_133934/, 20220410_033519/, ...)
│   │                             
│   ├── mask/                     # Segmentation masks for evaluation
│   └── metadata/                 # Metadata files
│
├── WebMaterial/
│   ├── train/                    # Training set (organized in section 1.1, Images organized by material categories)
│   │   └── image/                
│   ├── test/                     # Evaluation set (organized in section 1.1, Images organized by material categories)
│   │   ├── image/                
│   │   └── mask/                 # Corresponding segmentation masks
│   ├── metadata/                 # Metadata files
│   └── url/                      # Download URLs and reorganization paths for section 1.1
│
└── OpenSurfaces/
    ├── image/                    # Evaluation images
    ├── mask/                     # Evaluation segmentation masks
    └── metadata/                 # Metadata files
```


