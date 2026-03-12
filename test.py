import cv2
import numpy as np
import random
import torch
from train import *
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.classification import Accuracy, F1Score, ConfusionMatrix
from sklearn.metrics import f1_score, roc_auc_score

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    return parser.parse_args()

def main():
    args = get_args()
    config = py2cfg(args.config_path)
    weights_path = "D:/airs/ESA_CLIP/model_weights/{}".format(config.weights_name)
    model = CLIP_Train.load_from_checkpoint(
        os.path.join(weights_path, config.test_weights_name + '.ckpt'), config=config)

    model.to('cuda')
    model.eval()
    overall_acc = Accuracy(task="multiclass", num_classes=config.num_classes, average='micro')
    acc = Accuracy(task="multiclass", num_classes=config.num_classes, average='none')
    # macro_f1 = F1Score(task="multiclass", num_classes=config.num_classes, average='macro')
    # micro_f1 = F1Score(task="multiclass", num_classes=config.num_classes, average='micro')
    confmat = ConfusionMatrix(task="multiclass", num_classes=config.num_classes)

    anomaly_classes = ['blue algae', 'bushfire', 'debris flow', 'farmland fire',
                      'flood', 'forest fire', 'green tide', 'red tide', 'volcanic eruption']
    normal_class = 'normal'
    anomaly_indices = [config.classes.index(cls) for cls in anomaly_classes]
    normal_index = config.classes.index(normal_class)

    all_binary_gt = []
    all_pos_probs = []
    all_binary_preds = []

    test_dataset = MSESADDataset(mode='test')
    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=16,
            pin_memory=True,
            drop_last=False,
            num_workers=8,
            persistent_workers=True,
        )
        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            res = model(input['img_rgb'].to("cuda"), input['img_swir'].to("cuda"))
            gt = input['cls']
            logits = res['logits'].cpu()
            # preds = torch.argmax(logits, dim=1)
            acc.update(logits, gt)
            overall_acc.update(logits, gt)
            # # macro_f1.update(logits, gt)
            # # micro_f1.update(logits, gt)
            # # confmat.update(preds, gt)

            probs = F.softmax(res['logits'], dim=1).cpu()
            pos_probs = probs[:, anomaly_indices].sum(dim=1)
            binary_gt = torch.isin(gt, torch.tensor(anomaly_indices)).long()

            # 收集数据
            all_binary_gt.extend(binary_gt.tolist())
            all_pos_probs.extend(pos_probs.tolist())
            all_binary_preds.extend((pos_probs >= 0.5).long().tolist())

        # 计算指标
    binary_gt_tensor = torch.tensor(all_binary_gt)
    binary_preds_tensor = torch.tensor(all_binary_preds)
    pos_probs_tensor = torch.tensor(all_pos_probs)

    TP = ((binary_preds_tensor == 1) & (binary_gt_tensor == 1)).sum().item()
    TN = ((binary_preds_tensor == 0) & (binary_gt_tensor == 0)).sum().item()
    FP = ((binary_preds_tensor == 1) & (binary_gt_tensor == 0)).sum().item()
    FN = ((binary_preds_tensor == 0) & (binary_gt_tensor == 1)).sum().item()

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1 = f1_score(all_binary_gt, all_binary_preds)
    auc = roc_auc_score(all_binary_gt, all_pos_probs)
    print(TP)
    print(TN)
    print(FP)
    print(FN)

    # 打印结果
    print(config.weights_name)
    print(f"正样本召回率: {recall *100:.4f}")
    print(f"负样本特异度: {specificity*100:.4f}")
    print(f"F1分数: {f1*100:.4f}")
    print(f"AUC值: {auc:.4f}")

    # print(config.weights_name)
    # # print(f'Micro F1 Score: {micro_f1.compute().item():.8f}')
    # # print(f'Macro F1 Score: {macro_f1.compute().item() * 100:.2f}')
    #
    print(f'Overall Accuracy: {overall_acc.compute().item() * 100:.3f}')
    per_class_acc_list = []
    for c, a in zip(config.classes, acc.compute().cpu().numpy()):
        per_class_acc_list.append((c,a))
    print(per_class_acc_list)
    #
    # cm = confmat.compute()
    # normalized_cm = cm / cm.sum(dim=1, keepdim=True)
    # formatted_cm = np.around(normalized_cm.cpu().numpy(), decimals=3)
    # print(formatted_cm)

if __name__ == "__main__":
    main()
