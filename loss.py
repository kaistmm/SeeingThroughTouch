from itertools import combinations
from pathlib import Path
import torch 
import torch.nn as nn
import torch.nn.functional as F
from STT import ModalityType
from timm.utils import accuracy

PROJECT_ROOT = Path(__file__).resolve().parent
TOUCH_AND_GO_METADATA_DIR = PROJECT_ROOT / "datasets" / "touch_and_go" / "metadata"

def construct_top_k_mask(affinity_matrix, k=5):
    topk_mat = torch.topk(affinity_matrix, k=k, dim=1)[1]
    topk_bool = torch.zeros_like(affinity_matrix, dtype=torch.bool)
    topk_bool.scatter_(1, topk_mat, True)
    return topk_bool

def compute_category_accuracy(affinity_matrix, category_index, topk=(1, 5)):
    device = affinity_matrix.device
    category_index = category_index.to(device)
    topk_indices = affinity_matrix.topk(max(topk), dim=1).indices  
    results = []
    for k in topk:
        topk_k = topk_indices[:, :k]             
        retrieved_cats = category_index[topk_k]  
        query_cats = category_index.unsqueeze(1).expand_as(retrieved_cats)  
        acc = (retrieved_cats == query_cats).any(dim=1).float().mean() * 100
        results.append(acc)
    return results

def get_category_labels(test_label_path):
        category_list = []
        with open(test_label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    id, category = line.split(',')
                    ### SKIP WEIRD DATA from Touch_and_Go (only have gelsight frames)
                    if ("20220318_020426" in id) or ("20220318_021048" in id):
                        continue
                    else:
                        category_list.append(int(category))
                except ValueError:
                    print(f"Skipped malformed line: {line}")
                    continue

        return torch.tensor(category_list, dtype=torch.long)

class VisuoTactileLoss(nn.Module):
    def __init__(
        self,
        active_modalities=[ModalityType.VISION, ModalityType.TACTILE],
        lang_am_temp=0.05, 
        similarity_thres=0.9,
        category_match=False,
        aggregation_pool='max',  
        test_split_type="no_inter",
        use_aggregation_loss=True,  
    ):

        super(VisuoTactileLoss, self).__init__()
        self.active_modalities = active_modalities 
        self.lang_am_temp = lang_am_temp
        self.similarity_thres = similarity_thres
        self.aggregation_pool = aggregation_pool
        self.test_split_type = test_split_type
        self.use_aggregation_loss = use_aggregation_loss
        if self.use_aggregation_loss:
            print(f"[INFO] Using Aggregation Loss")
            print(f"[INFO] Aggregation Pool: {self.aggregation_pool}")
        else:
            print(f"[INFO] Using Standard CLIP Loss")
        print(f"[INFO] Test Split Type: {self.test_split_type}")
        
        self.category_match = category_match
        if self.category_match:
            print(f"[INFO] Using Category Matching for Evaluation")
        else:
            print(f"[INFO] Using Samplewise Matching for Evaluation")
        if self.category_match:
            # Select test_split_type
            if test_split_type == "no_inter": # Leakage-free
                test_file = str(TOUCH_AND_GO_METADATA_DIR / 'test_1118_touch_instances.txt')
            else:  # "original"
                test_file = str(TOUCH_AND_GO_METADATA_DIR / 'test_1113.txt')
            self.category_labels = get_category_labels(test_file)

    def clip_loss(self, class_a_feat, class_b_feat, logit_scale):
        labels = torch.arange(class_a_feat.shape[0], device=class_a_feat.device, dtype=torch.long)
        class_a_feat = class_a_feat.reshape(class_a_feat.shape[0], -1)  
        class_b_feat = class_b_feat.reshape(class_b_feat.shape[0], -1)
        affinity_matrix = logit_scale * class_a_feat @ class_b_feat.T
        row_loss = F.cross_entropy(affinity_matrix, labels)
        col_loss = F.cross_entropy(affinity_matrix.T, labels)
        return (row_loss + col_loss) / 2, affinity_matrix

    def clip_loss_aggregation(self, class_a_feat, class_b_feat, logit_scale,pool='max'):
        labels = torch.arange(class_a_feat.shape[0], device=class_a_feat.device, dtype=torch.long)
        B,Ha,Wa,D = class_a_feat.shape
        B,Hb,Wb,D = class_b_feat.shape
        
        class_a_feat = class_a_feat.reshape(B,Ha*Wa,D) 
        class_b_feat = class_b_feat.reshape(B,Hb*Wb,D) 
        
        class_a_feat = class_a_feat.mean(dim=1, keepdim=True) 
        sim_heatmap_matrix = torch.einsum("bnd,kmd->bkm", class_a_feat, class_b_feat) 
        sim_heatmap_matrix = sim_heatmap_matrix * logit_scale 

        if pool == 'mean':
            affinity_matrix = sim_heatmap_matrix.mean(dim=-1)
        elif pool == 'max':
            affinity_matrix = sim_heatmap_matrix.max(dim=-1)[0]
        else:
            raise ValueError(f"Unsupported pooling method: {pool}")
        
        row_loss = F.cross_entropy(affinity_matrix, labels)
        col_loss = F.cross_entropy(affinity_matrix.T, labels)
        return (row_loss + col_loss) / 2, affinity_matrix
    
    def get_acc_from_affinity(self, affinity_matrix, gt_distribution=None):
        if gt_distribution is not None:
            positive_mask = gt_distribution > self.similarity_thres

            top1_bool = construct_top_k_mask(affinity_matrix, k=1)
            top5_bool = construct_top_k_mask(affinity_matrix, k=5)
            
            top1_bool_t = construct_top_k_mask(affinity_matrix.T, k=1)
            top5_bool_t = construct_top_k_mask(affinity_matrix.T, k=5)

            acc1 = torch.any(top1_bool & positive_mask, dim=1).float().mean()
            acc5 = torch.any(top5_bool & positive_mask, dim=1).float().mean()
            acc1_t = torch.any(top1_bool_t & positive_mask, dim=1).float().mean() 
            acc5_t = torch.any(top5_bool_t & positive_mask, dim=1).float().mean() 
            acc1 = (acc1 + acc1_t) / 2 * 100 
            acc5 = (acc5 + acc5_t) / 2 * 100 
            return acc1, acc5
        
        if self.category_match and affinity_matrix.shape[0] == len(self.category_labels):
            acc1, acc5, acc10 = compute_category_accuracy(affinity_matrix, self.category_labels, topk=(1, 5, 10))
            acc1_t, acc5_t, acc10_t = compute_category_accuracy(affinity_matrix.T, self.category_labels, topk=(1, 5, 10))
            return acc1, acc5, acc10, acc1_t, acc5_t, acc10_t

        labels = torch.arange(affinity_matrix.shape[0], device=affinity_matrix.device, dtype=torch.long)
        acc1, acc5, acc10 = accuracy(affinity_matrix, labels, topk=(1, 5, 10))
        acc1_t, acc5_t, acc10_t = accuracy(affinity_matrix.T, labels, topk=(1, 5, 10))
        return acc1, acc5, acc10, acc1_t, acc5_t, acc10_t
        

    def forward(self, feature_dict : dict, logit_scale : torch.Tensor, output_dict=False):
        total_loss = 0
        losses = {}
        class_pairs = list(combinations(self.active_modalities, 2))
        for class_a, class_b in class_pairs:
            class_a, class_b = sorted([class_a, class_b])
            class_a_feat = feature_dict[class_a]
            class_b_feat = feature_dict[class_b]
            
            if self.use_aggregation_loss:
                loss, affinity_mat = self.clip_loss_aggregation(class_a_feat, class_b_feat, logit_scale, self.aggregation_pool)
            else:
                loss, affinity_mat = self.clip_loss(class_a_feat, class_b_feat, logit_scale)
        
            acc1, acc5, acc10, acc1_t, acc5_t, acc10_t = self.get_acc_from_affinity(affinity_mat)

            losses[f"{class_a}_{class_b}"] = loss
            losses[f"{class_a}_{class_b}_acc1"] = acc1
            losses[f"{class_a}_{class_b}_acc5"] = acc5
            losses[f"{class_a}_{class_b}_acc10"] = acc10
            losses[f"{class_b}_{class_a}_acc1"] = acc1_t
            losses[f"{class_b}_{class_a}_acc5"] = acc5_t
            losses[f"{class_b}_{class_a}_acc10"] = acc10_t

            total_loss += loss

        losses["average_loss"] = total_loss / len(class_pairs)
        losses["clip_agg_loss"] = loss / len(class_pairs)
        losses["average_acc1"] = torch.mean(torch.stack([i for k,i in losses.items() if (("acc1" in k) and ("acc10" not in k))]))
        losses["average_acc5"] = torch.mean(torch.stack([i for k,i in losses.items() if "acc5" in k]))
        losses["average_acc10"] = torch.mean(torch.stack([i for k,i in losses.items() if "acc10" in k]))
        return losses if output_dict else losses["average_loss"]
