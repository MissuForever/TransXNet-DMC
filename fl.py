import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import time
from thop import profile
from fvcore.nn import FlopCountAnalysis
from sklearn.metrics import roc_curve, auc
import random

#三个backbone
from models.edgenextsmall import edgenext_xx_small
from models.repvit import repvit_m0_9


# 配置参数
class Config:
    data_dir = "./Dataset"  # 数据集根目录
    ct_dir = os.path.join(data_dir, "Brain Tumor CT scan Images")
    mri_dir = os.path.join(data_dir, "Brain Tumor MRI images")
    batch_size = 4
    num_workers = 4
    num_epochs = 30
    learning_rate = 7e-6
    input_size = 256  # 匹配原始图像尺寸
    num_classes = 2  # 二分类：Healthy vs Tumor
    class_names = ['Healthy', 'Tumor']
    # 可选: repvit, edgenext, transxnet_base, transxnet_mca, transxnet_mca_mudd, transxnet_dmc
    backbone_name = 'repvit'


def build_backbone(name: str) -> nn.Module:
    """按名称构建backbone。TransXNet分支使用惰性导入，避免无关依赖导致启动失败。"""
    if name == 'repvit':
        return repvit_m0_9()
    if name == 'edgenext':
        return edgenext_xx_small(classifier_dropout=0.0)
    if name == 'transxnet_base':
        from models.transxnet import transxnet_t as build_fn
        return build_fn(num_classes=1000, img_size=Config.input_size)
    if name == 'transxnet_mca':
        from models.transxnet_mca import transxnet_t as build_fn
        return build_fn(num_classes=1000, img_size=Config.input_size)
    if name == 'transxnet_mca_mudd':
        from models.transxnet_mca_mudd import transxnet_t as build_fn
        return build_fn(num_classes=1000, img_size=Config.input_size)
    if name == 'transxnet_dmc':
        from models.transxnet_dmc import transxnet_t as build_fn
        return build_fn(num_classes=1000, img_size=Config.input_size)
    raise ValueError(f"Unsupported backbone_name: {name}")


def set_seed(seed=42):
    """固定随机种子确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 多模态自定义数据集
class MultiModalDataset(Dataset):
    def __init__(self, ct_data_dict, mri_data_dict, transform=None):
        """
        ct_data_dict: CT图像的数据字典 {class_name: [image_paths]}
        mri_data_dict: MRI图像的数据字典 {class_name: [image_paths]}
        """
        self.ct_image_paths = []
        self.mri_image_paths = []
        self.labels = []
        self.transform = transform

        # 确保CT和MRI数据按相同顺序配对
        for class_idx, class_name in enumerate(Config.class_names):
            ct_paths = ct_data_dict.get(class_name, [])
            mri_paths = mri_data_dict.get(class_name, [])

            # 取最小长度确保配对
            min_length = min(len(ct_paths), len(mri_paths))

            if min_length == 0:
                print(f"警告: 类别 {class_name} 缺少CT或MRI数据")
                continue

            # 按顺序配对
            for i in range(min_length):
                self.ct_image_paths.append(ct_paths[i])
                self.mri_image_paths.append(mri_paths[i])
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.ct_image_paths)

    def __getitem__(self, idx):
        # 加载CT图像
        ct_image = Image.open(self.ct_image_paths[idx]).convert('RGB')
        # 加载MRI图像
        mri_image = Image.open(self.mri_image_paths[idx]).convert('RGB')

        # 应用数据增强
        if self.transform:
            ct_image = self.transform(ct_image)
            mri_image = self.transform(mri_image)

        return (ct_image, mri_image), self.labels[idx]


# 多模态模型
class MultiModalHighResMamba(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # 两个独立的backbone，分别处理CT和MRI
        self.ct_backbone = build_backbone(Config.backbone_name)
        self.mri_backbone = build_backbone(Config.backbone_name)


        # 分类头 - 输入维度为2000 (1000 + 1000)
        self.head = nn.Sequential(
            nn.Linear(2000, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, ct_x, mri_x):
        # 分别提取CT和MRI特征
        ct_features = self.ct_backbone(ct_x)
        mri_features = self.mri_backbone(mri_x)

        # 拼接特征
        fused_features = torch.cat([ct_features, mri_features], dim=1)

        # 分类
        return self.head(fused_features)


# 数据准备函数
def prepare_multimodal_datasets(seed=42):
    set_seed(seed)

    # 准备CT数据
    ct_data_dict = defaultdict(list)
    for class_name in Config.class_names:
        class_dir = os.path.join(Config.ct_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"警告: CT目录不存在: {class_dir}")
            continue

        # 收集所有CT图片文件
        image_files = []
        for file in sorted(os.listdir(class_dir)):  # 按文件名排序确保顺序一致
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(class_dir, file)
                image_files.append(file_path)

        ct_data_dict[class_name] = sorted(image_files)  # 确保顺序一致

    # 准备MRI数据
    mri_data_dict = defaultdict(list)
    for class_name in Config.class_names:
        class_dir = os.path.join(Config.mri_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"警告: MRI目录不存在: {class_dir}")
            continue

        # 收集所有MRI图片文件
        image_files = []
        for file in sorted(os.listdir(class_dir)):  # 按文件名排序确保顺序一致
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(class_dir, file)
                image_files.append(file_path)

        mri_data_dict[class_name] = sorted(image_files)  # 确保顺序一致

    # 检查数据平衡性
    for class_name in Config.class_names:
        ct_count = len(ct_data_dict.get(class_name, []))
        mri_count = len(mri_data_dict.get(class_name, []))
        print(f"类别 {class_name}: CT图像 {ct_count}张, MRI图像 {mri_count}张")

    # 分层划分训练测试集
    train_ct_data = defaultdict(list)
    test_ct_data = defaultdict(list)
    train_mri_data = defaultdict(list)
    test_mri_data = defaultdict(list)

    for class_name in Config.class_names:
        ct_paths = ct_data_dict.get(class_name, [])
        mri_paths = mri_data_dict.get(class_name, [])

        # 确保CT和MRI数据长度一致
        min_length = min(len(ct_paths), len(mri_paths))
        if min_length == 0:
            continue

        ct_paths = ct_paths[:min_length]
        mri_paths = mri_paths[:min_length]

        # 随机打乱但保持CT和MRI对应关系
        indices = np.random.permutation(min_length)
        ct_paths = [ct_paths[i] for i in indices]
        mri_paths = [mri_paths[i] for i in indices]

        split = int(0.8 * min_length)
        train_ct_data[class_name] = ct_paths[:split]
        test_ct_data[class_name] = ct_paths[split:]
        train_mri_data[class_name] = mri_paths[:split]
        test_mri_data[class_name] = mri_paths[split:]

    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize(Config.input_size),
        transforms.CenterCrop(Config.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(3),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(Config.input_size),
        transforms.CenterCrop(Config.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return {
        'train': MultiModalDataset(train_ct_data, train_mri_data, train_transform),
        'test': MultiModalDataset(test_ct_data, test_mri_data, test_transform)
    }


def train_multimodal_model_amp(model, dataloaders, criterion, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler()
    history = defaultdict(list)
    best_acc = 0.0
    best_model_path = 'best_multimodal_model.pth'

    for epoch in range(Config.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        with tqdm(dataloaders['train'], desc=f'Train Epoch {epoch + 1}') as pbar:
            for (ct_inputs, mri_inputs), labels in pbar:
                ct_inputs = ct_inputs.to(device)
                mri_inputs = mri_inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 混合精度训练
                with torch.cuda.amp.autocast():
                    outputs = model(ct_inputs, mri_inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(optimizer)
                scaler.update()

                # 学习率调度
                lr_ = Config.learning_rate * 0.5 * (1 + np.cos(np.pi * epoch / Config.num_epochs))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({'Loss': loss.item(), 'Acc': correct / total})

        train_acc = correct / total
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss / len(dataloaders['train']))

        # 验证阶段
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        model.eval()

        with torch.no_grad(), tqdm(dataloaders['test'], desc='Validating') as pbar:
            for (ct_inputs, mri_inputs), labels in pbar:
                ct_inputs = ct_inputs.to(device)
                mri_inputs = mri_inputs.to(device)
                labels = labels.to(device)

                outputs = model(ct_inputs, mri_inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({'Val_Loss': loss.item(), 'Val_Acc': val_correct / val_total})

        val_acc = val_correct / val_total
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss / len(dataloaders['test']))

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with accuracy: {best_acc:.4f}")

    print('trainacc', history['train_acc'])
    print('trainloss', history['train_loss'])
    print('valacc', history['val_acc'])
    print('valloss', history['val_loss'])
    return history


# #可视化
def plot_roc_curve(test_loader, model, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    y_true = []
    y_scores = []

    with torch.no_grad():
        for (ct_inputs, mri_inputs), labels in test_loader:  # 修改这里
            ct_inputs = ct_inputs.to(device)  # 分别处理CT和MRI输入
            mri_inputs = mri_inputs.to(device)
            labels = labels.to(device)

            outputs = model(ct_inputs, mri_inputs)  # 修改这里，传入两个输入
            probabilities = torch.softmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_scores.extend(probabilities.cpu().numpy())

    # 二分类处理
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # 二分类只需要一个类别的概率
    fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')

    return roc_auc


def enhanced_multimodal_visualization(history, test_loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载最佳模型
    best_model_path = 'best_multimodal_model.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("✓ 已加载最佳多模态模型进行可视化")
    else:
        print("⚠ 未找到最佳模型文件，使用最终模型")

    model = model.to(device)
    model.eval()

    # 绘制训练曲线
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('multimodal_training_curves.png')


    # 生成混淆矩阵和收集预测结果
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for (ct_inputs, mri_inputs), labels in test_loader:
            ct_inputs = ct_inputs.to(device)
            mri_inputs = mri_inputs.to(device)
            labels = labels.to(device)

            outputs = model(ct_inputs, mri_inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    # 转换为numpy数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 计算基本准确率
    accuracy = (all_preds == all_labels).mean()

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 计算各项指标
    print("\n" + "=" * 80)
    print("多模态模型评估指标")
    print("=" * 80)

    # 计算每个类别的指标
    num_classes = len(Config.class_names)
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    support_per_class = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
        support_per_class.append(support)

        print(f"类别 {Config.class_names[i]}:")
        print(f"  - 精确度 (Precision): {precision:.4f}")
        print(f"  - 召回率 (Recall):    {recall:.4f}")
        print(f"  - F1分数:             {f1:.4f}")
        print(f"  - 支持度:             {support}")
        print()

    # 计算宏平均和加权平均
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)
    weighted_precision = np.average(precision_per_class, weights=support_per_class)
    weighted_recall = np.average(recall_per_class, weights=support_per_class)
    weighted_f1 = np.average(f1_per_class, weights=support_per_class)
    uar = macro_recall
    uf1 = macro_f1

    print("整体指标:")
    print(f"准确率 (Accuracy):        {accuracy:.4f}")
    print(f"宏平均精确度:             {macro_precision:.4f}")
    print(f"宏平均召回率 (UAR):       {uar:.4f}")
    print(f"宏平均F1分数 (UF1):       {uf1:.4f}")
    print(f"加权平均精确度:           {weighted_precision:.4f}")
    print(f"加权平均召回率:           {weighted_recall:.4f}")
    print(f"加权平均F1分数:           {weighted_f1:.4f}")

    # 保存指标到文件
    with open('multimodal_evaluation_metrics.txt', 'w', encoding='utf-8') as f:
        f.write("多模态模型评估指标\n")
        f.write("=" * 50 + "\n")
        f.write(f"准确率 (Accuracy): {accuracy:.4f}\n")
        f.write(f"宏平均精确度: {macro_precision:.4f}\n")
        f.write(f"宏平均召回率 (UAR): {uar:.4f}\n")
        f.write(f"宏平均F1分数 (UF1): {uf1:.4f}\n")
        f.write(f"加权平均精确度: {weighted_precision:.4f}\n")
        f.write(f"加权平均召回率: {weighted_recall:.4f}\n")
        f.write(f"加权平均F1分数: {weighted_f1:.4f}\n")

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.class_names,
                yticklabels=Config.class_names)
    plt.title(f'Multimodal Confusion Matrix (Accuracy: {accuracy:.4f})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('multimodal_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    roc_auc = plot_roc_curve(test_loader, model, Config.class_names)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    print(f"多模态模型测试集准确率: {accuracy:.4f}")
    print(f"测试集最佳准确率: {max(history['val_acc']):.4f}")
    print(f"UAR (宏平均召回率): {uar:.4f}")
    print(f"UF1 (宏平均F1分数): {uf1:.4f}")

    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'uar': uar,
        'uf1': uf1,
        'confusion_matrix': cm
    }


def analyze_multimodal_model(model, input_size=(1, 3, 256, 256)):
    """分析多模态模型的参数量和计算量"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 创建随机输入（CT和MRI）
    dummy_ct = torch.randn(input_size).to(device)
    dummy_mri = torch.randn(input_size).to(device)

    # 计算FLOPs和参数量
    def count_operations(model, ct_input, mri_input):
        # CT backbone
        ct_flops = FlopCountAnalysis(model.ct_backbone, ct_input)
        # MRI backbone
        mri_flops = FlopCountAnalysis(model.mri_backbone, mri_input)
        # 分类头（近似计算）
        dummy_features = torch.randn(1, 2000).to(device)
        head_flops = FlopCountAnalysis(model.head, dummy_features)

        total_flops = ct_flops.total() + mri_flops.total() + head_flops.total()

        # 参数量
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return total_params, total_flops

    params, flops = count_operations(model, dummy_ct, dummy_mri)

    print("=" * 50)
    print("多模态模型分析结果:")
    print(f"总参数量: {params / 1e6:.2f} M")
    print(f"总FLOPs: {flops / 1e9:.2f} G")
    print("=" * 50)

    return params, flops


# 主流程
def main():
    set_seed(42)

    # 准备多模态数据
    print("准备多模态数据集...")
    datasets = prepare_multimodal_datasets(seed=42)

    dataloaders = {
        'train': DataLoader(datasets['train'],
                            batch_size=Config.batch_size,
                            shuffle=True,
                            num_workers=Config.num_workers,
                            pin_memory=True,
                            drop_last=True),
        'test': DataLoader(datasets['test'],
                           batch_size=Config.batch_size,
                           shuffle=False,
                           num_workers=Config.num_workers,
                           pin_memory=True,
                           drop_last=True)
    }

    # 初始化多模态模型
    model = MultiModalHighResMamba(num_classes=Config.num_classes)

    # 模型分析
    print("开始多模态模型分析...")
    params, flops = analyze_multimodal_model(model)

    if torch.cuda.is_available():
        model = model.cuda()

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=Config.learning_rate,
                            weight_decay=1e-4,
                            betas=(0.9, 0.999))

    # 训练多模态模型
    print("开始训练多模态模型...")
    history = train_multimodal_model_amp(model, dataloaders, criterion, optimizer)

    # 结果可视化
    enhanced_multimodal_visualization(history, dataloaders['test'], model)

    # 保存最终模型
    torch.save(model.state_dict(), 'final_multimodal_model.pth')
    print("多模态模型训练完成！")


if __name__ == '__main__':
    main()