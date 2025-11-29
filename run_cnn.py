import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn.manifold import TSNE
from PIL import Image
import os
from tqdm import tqdm
import time
import csv
import copy
import argparse


# 1. 参数配置 (Hyper-parameters)
DATA_PATH = './dataset'  # 数据集根目录
BATCH_SIZE = 32          # 批次大小
LEARNING_RATE = 0.001    # 学习率
EPOCHS = 15              # 训练轮数
NUM_CLASSES = 27         # PlantDoc数据集类别数 [cite: 9]
TARGET_VAL_ACC = 0.48    # 收敛阈值（验证准确率）

# 检测设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {DEVICE}")

parser = argparse.ArgumentParser(description='基于PlantDoc的植物病害识别训练脚本，支持骨干选择、数据增强与收敛阈值配置')
parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18','resnet34','resnet50','mobilenet_v2'], help='选择骨干网络：resnet18/resnet34/resnet50/mobilenet_v2，默认resnet18')
parser.add_argument('--target_acc', type=float, default=0.48, help='验证准确率收敛阈值，用于统计达到阈值所需轮次，默认0.48')
parser.add_argument('--epochs', type=int, default=EPOCHS, help='训练轮数，默认当前脚本配置')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='批大小，默认当前脚本配置')
parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='学习率，默认当前脚本配置')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam','sgd','rmsprop','adamw'], help='优化器类型：adam/sgd/rmsprop/adamw，默认adam')
parser.add_argument('--augment', type=int, choices=[0,1], default=0, help='是否开启训练集数据增强：1开启，0关闭，默认0')
parser.add_argument('--pretrained', type=int, choices=[0,1], default=1, help='是否使用预训练权重：1使用，0不使用，默认1')
parser.add_argument('--scheduler', type=str, default='none', choices=['none','step','cosine','plateau'], help='学习率调度器：none/step/cosine/plateau，默认none')
parser.add_argument('--step_size', type=int, default=5, help='StepLR 的 step_size，默认5')
parser.add_argument('--gamma', type=float, default=0.1, help='StepLR/Plateau 的衰减因子 gamma，默认0.1')
parser.add_argument('--early_stop', type=int, choices=[0,1], default=0, help='是否启用早停：1启用，0关闭，默认0')
parser.add_argument('--patience', type=int, default=5, help='早停 patience 轮数，默认5')
parser.add_argument('--min_delta', type=float, default=0.0, help='早停最小改进幅度 min_delta，默认0.0')
parser.add_argument('--weight_decay', type=float, default=0.0, help='优化器的权重衰减（L2），默认0.0')
parser.add_argument('--freeze', type=str, default='full', choices=['full','head','partial'], help='参数冻结策略：full/head/partial，默认full')
parser.add_argument('--loss', type=str, default='ce', choices=['ce','focal'], help='损失函数：ce 或 focal，默认ce')
parser.add_argument('--class_weight', type=int, choices=[0,1], default=0, help='是否启用类别权重用于不均衡：1启用，0关闭，默认0')
parser.add_argument('--gamma_focal', type=float, default=2.0, help='FocalLoss 的 gamma，默认2.0')
parser.add_argument('--load_ckpt', type=str, default='', help='加载已有模型权重路径')
parser.add_argument('--output_dir', type=str, default='', help='覆盖默认输出目录')
parser.add_argument('--visualize', type=int, choices=[0,1], default=0, help='启用中间输出可视化')
parser.add_argument('--vis_max_samples', type=int, default=12, help='Grad-CAM示例数量上限')
parser.add_argument('--top_confusions_csv', type=str, default='', help='指定易混样例CSV路径')
parser.add_argument('--input_size', type=int, default=224, help='输入分辨率（正方形边长）')
parser.add_argument('--jitter', type=float, default=0.1, help='颜色抖动强度系数')
parser.add_argument('--rotation', type=float, default=15.0, help='随机旋转角度范围')
parser.add_argument('--use_rrc', type=int, choices=[0,1], default=0, help='训练集使用RandomResizedCrop')
parser.add_argument('--save_preds', type=int, choices=[0,1], default=0, help='导出指定划分的预测为CSV')
parser.add_argument('--pred_split', type=str, default='val', choices=['train','val'], help='导出预测的划分')
parser.add_argument('--preds_csv', type=str, default='', help='预测CSV保存路径')
args = parser.parse_args()
BACKBONE_NAME = args.backbone
TARGET_VAL_ACC = args.target_acc
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
AUGMENT = bool(args.augment)
PRETRAINED = bool(args.pretrained)
OPTIMIZER_NAME = args.optimizer
SCHEDULER_NAME = args.scheduler
STEP_SIZE = args.step_size
GAMMA = args.gamma
EARLY_STOP = bool(args.early_stop)
PATIENCE = args.patience
MIN_DELTA = args.min_delta
WEIGHT_DECAY = args.weight_decay
FREEZE_MODE = args.freeze
LOSS_TYPE = args.loss
CLASS_WEIGHT_ENABLED = bool(args.class_weight)
GAMMA_FOCAL = args.gamma_focal
INPUT_SIZE = int(args.input_size)
JITTER_STRENGTH = float(args.jitter)
ROTATION_DEG = float(args.rotation)
USE_RRC = bool(args.use_rrc)
OUTPUT_DIR =  time.strftime("%Y%m%d_%H%M%S") + f"_{BACKBONE_NAME}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_opt{OPTIMIZER_NAME}_ep{EPOCHS}_batch{BATCH_SIZE}_aug{AUGMENT}_sch{SCHEDULER_NAME}_wd{WEIGHT_DECAY}_es{EARLY_STOP}_frz{FREEZE_MODE}_loss{LOSS_TYPE}_cw{CLASS_WEIGHT_ENABLED}"
if args.output_dir:
    OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)
RUN_ID = OUTPUT_DIR
LOG_CSV_PATH = os.path.join(OUTPUT_DIR, 'training_log.csv')


# 2. 数据准备 (Data Preparation)

# 定义数据预处理和增强
if AUGMENT:
    aug_ops = []
    if USE_RRC:
        aug_ops.append(transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)))
    else:
        aug_ops.append(transforms.Resize((INPUT_SIZE, INPUT_SIZE)))
    aug_ops.append(transforms.RandomHorizontalFlip())
    aug_ops.append(transforms.RandomRotation(ROTATION_DEG))
    if JITTER_STRENGTH > 0.0:
        aug_ops.append(transforms.ColorJitter(brightness=JITTER_STRENGTH, contrast=JITTER_STRENGTH, saturation=JITTER_STRENGTH))
    aug_ops.append(transforms.ToTensor())
    aug_ops.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    train_tfms = transforms.Compose(aug_ops)
else:
    train_tfms = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
data_transforms = {
    'train': train_tfms,
    'val': transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Loading data...")
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(DATA_PATH, 'TRAIN'), data_transforms['train']),
    'val': datasets.ImageFolder(os.path.join(DATA_PATH, 'TEST'), data_transforms['val'])
}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(f"Train images: {dataset_sizes['train']}")
print(f"Test/Val images: {dataset_sizes['val']}")
print(f"Classes detected: {len(class_names)}")

# 3. 模型构建 (Model Setup)

def create_model(name, num_classes, pretrained=True):
    if name == 'resnet18':
        m = models.resnet18(pretrained=pretrained)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
    elif name == 'resnet34':
        m = models.resnet34(pretrained=pretrained)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
    elif name == 'resnet50':
        m = models.resnet50(pretrained=pretrained)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
    elif name == 'mobilenet_v2':
        m = models.mobilenet_v2(pretrained=pretrained)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    return m

model = create_model(BACKBONE_NAME, NUM_CLASSES, PRETRAINED)


model = model.to(DEVICE)

def apply_freeze(m, backbone, mode):
    if mode == 'full':
        for p in m.parameters():
            p.requires_grad = True
    elif mode == 'head':
        for p in m.parameters():
            p.requires_grad = False
        if backbone.startswith('resnet'):
            for p in m.fc.parameters():
                p.requires_grad = True
        elif backbone == 'mobilenet_v2':
            for p in m.classifier.parameters():
                p.requires_grad = True
    elif mode == 'partial':
        for p in m.parameters():
            p.requires_grad = False
        if backbone.startswith('resnet'):
            for p in m.layer4.parameters():
                p.requires_grad = True
            for p in m.fc.parameters():
                p.requires_grad = True
        elif backbone == 'mobilenet_v2':
            try:
                for p in m.features[-1].parameters():
                    p.requires_grad = True
            except Exception:
                pass
            for p in m.classifier.parameters():
                p.requires_grad = True

apply_freeze(model, BACKBONE_NAME, FREEZE_MODE)

def compute_class_weights(dataset):
    counts = np.zeros(len(dataset.classes), dtype=np.float32)
    for _, label in dataset.imgs:
        counts[label] += 1
    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights * (len(weights) / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('alpha', alpha if alpha is not None else None)
    def forward(self, logits, targets):
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        targets = targets.long()
        pt = probs.gather(1, targets.view(-1,1)).squeeze(1)
        log_pt = log_probs.gather(1, targets.view(-1,1)).squeeze(1)
        focal = (1 - pt) ** self.gamma
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            loss = -at * focal * log_pt
        else:
            loss = -focal * log_pt
        return loss.mean()

class_weights = None
if CLASS_WEIGHT_ENABLED:
    class_weights = compute_class_weights(image_datasets['train']).to(DEVICE)

if LOSS_TYPE == 'ce':
    criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    alpha = class_weights if CLASS_WEIGHT_ENABLED else None
    criterion = FocalLoss(gamma=GAMMA_FOCAL, alpha=alpha)

def create_optimizer(name, params, lr, weight_decay=0.0):
    if name == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

optimizer = create_optimizer(OPTIMIZER_NAME, model.parameters(), LEARNING_RATE, WEIGHT_DECAY)

def create_scheduler(name, optimizer):
    if name == 'none':
        return None
    if name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    if name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    if name == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=GAMMA, patience=2)
    raise ValueError(f"Unsupported scheduler: {name}")

scheduler = create_scheduler(SCHEDULER_NAME, optimizer)


# 4. 训练循环

def train_model(model, criterion, optimizer, num_epochs=10):
    since = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    best_val_acc = -float('inf')
    best_state_dict = None
    best_epoch = 0
    epoch_to_threshold = None
    patience_counter = 0
    with open(LOG_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'run_id','timestamp','backbone','pretrained','optimizer','learning_rate','batch_size','epochs',
            'criterion','device','num_classes','train_size','val_size','total_params','trainable_params','augment',
            'scheduler','weight_decay','early_stop','patience','min_delta','current_lr',
            'freeze','loss','class_weight',
            'epoch','train_loss','train_acc','val_loss','val_acc','best_val_acc','best_epoch','target_val_acc','epoch_to_threshold','stopped_epoch'
        ])

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        train_loss_epoch = None
        train_acc_epoch = None
        val_loss_epoch = None
        val_acc_epoch = None
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 进度条
            loop = tqdm(dataloaders[phase], desc=f"{phase}ing")
            
            for inputs, labels in loop:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # 更新进度条信息
                loop.set_postfix(loss=loss.item())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'train':
                train_loss_epoch = epoch_loss
                train_acc_epoch = epoch_acc.item()
            else:
                val_loss_epoch = epoch_loss
                val_acc_epoch = epoch_acc.item()
                if val_acc_epoch > best_val_acc:
                    best_val_acc = val_acc_epoch
                    best_state_dict = copy.deepcopy(model.state_dict())
                    best_epoch = epoch + 1
                if epoch_to_threshold is None and val_acc_epoch >= TARGET_VAL_ACC:
                    epoch_to_threshold = epoch + 1

        # Scheduler step
        if scheduler is not None:
            if SCHEDULER_NAME == 'plateau':
                scheduler.step(val_acc_epoch)
            else:
                scheduler.step()

        # Early stopping check
        stopped_epoch = ''
        if EARLY_STOP:
            if best_val_acc - val_acc_epoch > -MIN_DELTA:
                patience_counter += 1
            else:
                patience_counter = 0
            if patience_counter >= PATIENCE:
                stopped_epoch = str(epoch + 1)
                with open(LOG_CSV_PATH, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        RUN_ID,
                        time.strftime('%Y-%m-%d %H:%M:%S'),
                        BACKBONE_NAME,
                        PRETRAINED,
                        optimizer.__class__.__name__,
                        LEARNING_RATE,
                        BATCH_SIZE,
                        num_epochs,
                        criterion.__class__.__name__,
                        DEVICE.type,
                        NUM_CLASSES,
                        dataset_sizes['train'],
                        dataset_sizes['val'],
                        total_params,
                        trainable_params,
                        AUGMENT,
                        SCHEDULER_NAME,
                        WEIGHT_DECAY,
                        EARLY_STOP,
                        PATIENCE,
                        MIN_DELTA,
                        optimizer.param_groups[0]['lr'],
                        epoch+1,
                        train_loss_epoch,
                        train_acc_epoch,
                        val_loss_epoch,
                        val_acc_epoch,
                        best_val_acc,
                        best_epoch,
                        TARGET_VAL_ACC,
                        epoch_to_threshold,
                        stopped_epoch
                    ])
                print(f"Early stopping at epoch {epoch+1} (patience {PATIENCE})")
                break

        with open(LOG_CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                RUN_ID,
                time.strftime('%Y-%m-%d %H:%M:%S'),
                BACKBONE_NAME,
                PRETRAINED,
                optimizer.__class__.__name__,
                LEARNING_RATE,
                BATCH_SIZE,
                num_epochs,
                criterion.__class__.__name__,
                DEVICE.type,
                NUM_CLASSES,
                dataset_sizes['train'],
                dataset_sizes['val'],
                total_params,
                trainable_params,
                AUGMENT,
                SCHEDULER_NAME,
                WEIGHT_DECAY,
                EARLY_STOP,
                PATIENCE,
                MIN_DELTA,
                optimizer.param_groups[0]['lr'],
                FREEZE_MODE,
                LOSS_TYPE,
                CLASS_WEIGHT_ENABLED,
                epoch+1,
                train_loss_epoch,
                train_acc_epoch,
                val_loss_epoch,
                val_acc_epoch,
                best_val_acc,
                best_epoch,
                TARGET_VAL_ACC,
                epoch_to_threshold,
                ''
            ])

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return model, history, best_state_dict, best_val_acc, best_epoch, epoch_to_threshold

# 开始训练
trained_model = None
history = None
best_state_dict = None
best_val_acc = None
best_epoch = None
epoch_to_threshold = None
if args.load_ckpt:
    sd = torch.load(args.load_ckpt, map_location=DEVICE)
    model.load_state_dict(sd)
    trained_model = model
else:
    trained_model, history, best_state_dict, best_val_acc, best_epoch, epoch_to_threshold = train_model(model, criterion, optimizer, num_epochs=EPOCHS)

# ==========================
# 5. 结果可视化与保存 (Visualization)
# ==========================
# 5.1 保存训练好的模型
torch.save(trained_model.state_dict(), os.path.join(OUTPUT_DIR, f'{BACKBONE_NAME}.pth'))
print(f"Model saved to {os.path.join(OUTPUT_DIR, f'{BACKBONE_NAME}.pth')}\n")
if best_state_dict is not None:
    best_path = os.path.join(OUTPUT_DIR, f'{BACKBONE_NAME}_best.pth')
    torch.save(best_state_dict, best_path)
    print(f"Best model (epoch {best_epoch}, val_acc {best_val_acc:.4f}, target {TARGET_VAL_ACC:.2f}, epoch_to_threshold {epoch_to_threshold}) saved to {best_path}\n")

# 5.2 绘制 Loss 和 Accuracy 曲线 (用于作业报告的实验分析)
if history is not None:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_metrics.png'))
    print("Metrics plot saved as training_metrics.png")

# 5.3 混淆矩阵与失败样例分析 
def evaluate_performance(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Generating Confusion Matrix...")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 绘制热力图
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xticks(rotation=90)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    print("Confusion matrix saved as confusion_matrix.png")
    
    # 输出详细报告
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    fig = plt.figure(figsize=(12, max(6, len(class_names) * 0.35)))
    plt.axis('off')
    plt.text(0.01, 0.99, report, fontsize=10, va='top', family='monospace')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'classification_report.png'))
    plt.close(fig)

evaluate_performance(trained_model, dataloaders['val'])

def get_top_confusion_samples(csv_path, max_samples=12):
    samples = []
    if not os.path.isfile(csv_path):
        return samples
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(samples) >= max_samples:
                break
            samples.append({
                'true': row['true'],
                'pred': row['pred'],
                'file': row['example_file'],
                'prob': float(row.get('example_pred_prob', '0') or 0)
            })
    return samples

def preprocess_image_for_model(img_path):
    tfm = data_transforms['val']
    img = Image.open(img_path).convert('RGB')
    tensor = tfm(img).unsqueeze(0)
    return tensor

def grad_cam(model, inputs, target_layer, target_index):
    activations = {}
    gradients = {}
    def fwd_hook(module, inp, out):
        activations['value'] = out.detach()
    def bwd_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()
    handle_f = target_layer.register_forward_hook(fwd_hook)
    handle_b = target_layer.register_backward_hook(bwd_hook)
    model.zero_grad()
    outputs = model(inputs)
    score = outputs[:, target_index].sum()
    score.backward()
    acts = activations['value']
    grads = gradients['value']
    weights = grads.mean(dim=(2,3), keepdim=True)
    cam = (acts * weights).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = torch.nn.functional.interpolate(cam, size=inputs.shape[2:], mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    handle_f.remove()
    handle_b.remove()
    return cam

def visualize_grad_cam(model, samples, out_dir):
    model.eval()
    os.makedirs(os.path.join(out_dir, 'grad_cam'), exist_ok=True)
    if BACKBONE_NAME.startswith('resnet'):
        target_layer = model.layer4
    else:
        target_layer = None
    if target_layer is None:
        return
    for s in samples:
        img_path = s['file']
        x = preprocess_image_for_model(img_path).to(DEVICE)
        with torch.no_grad():
            y = model(x)
            pred_idx = int(torch.argmax(y, dim=1).item())
        cam = grad_cam(model, x, target_layer, pred_idx)
        arr = x.squeeze().cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
        img = (arr * std + mean)
        img = np.clip(img, 0, 1)
        img = np.transpose(img, (1,2,0))
        plt.figure(figsize=(6,5))
        plt.imshow(img)
        plt.imshow(cam, cmap='jet', alpha=0.45)
        plt.axis('off')
        base = os.path.basename(img_path)
        save_name = f"{s['true']}__pred_{s['pred']}__{base}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'grad_cam', save_name))
        plt.close()

def extract_features(model, loader):
    model.eval()
    feats = []
    labels = []
    backbone = nn.Sequential(*list(model.children())[:-1])
    with torch.no_grad():
        for inputs, lbls in loader:
            inputs = inputs.to(DEVICE)
            f = backbone(inputs)
            f = torch.flatten(f, 1)
            feats.append(f.cpu().numpy())
            labels.append(lbls.numpy())
    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    return feats, labels

def visualize_tsne_features(model, loader, class_names, out_dir):
    X, y = extract_features(model, loader)
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=min(30, max(5, len(y)//4)))
    Z = tsne.fit_transform(X)
    plt.figure(figsize=(12,10))
    scatter_colors = plt.cm.tab20(np.linspace(0,1,len(class_names)))
    for idx, cname in enumerate(class_names):
        mask = (y == idx)
        plt.scatter(Z[mask,0], Z[mask,1], s=12, color=scatter_colors[idx % len(scatter_colors)], label=cname, alpha=0.7)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=8)
    plt.title('t-SNE of penultimate features')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'features_tsne.png'))
    plt.close()

if args.visualize:
    csv_path = args.top_confusions_csv if args.top_confusions_csv else os.path.join(OUTPUT_DIR, 'top_confusions.csv')
    samples = get_top_confusion_samples(csv_path, max_samples=args.vis_max_samples)
    if len(samples) > 0:
        visualize_grad_cam(trained_model, samples, OUTPUT_DIR)
    visualize_tsne_features(trained_model, dataloaders['val'], class_names, OUTPUT_DIR)

def export_predictions(model, loader, class_names, out_csv):
    model.eval()
    files = [p for p,_ in loader.dataset.samples]
    labels = [l for _,l in loader.dataset.samples]
    idx = 0
    rows = []
    with torch.no_grad():
        for inputs, lbls in tqdm(loader, desc='predicting'):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            top1_prob, top1_idx = probs.max(dim=1)
            bsz = inputs.size(0)
            for j in range(bsz):
                f = files[idx+j]
                t_idx = labels[idx+j]
                p_idx = int(top1_idx[j].item())
                rows.append([
                    f,
                    t_idx,
                    class_names[t_idx],
                    p_idx,
                    class_names[p_idx],
                    float(top1_prob[j].item())
                ])
            idx += bsz
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file','true_idx','true_name','pred_idx','pred_name','pred_prob'])
        writer.writerows(rows)

if args.save_preds:
    split = args.pred_split
    export_loader = DataLoader(image_datasets[split], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    save_path = args.preds_csv if args.preds_csv else os.path.join(OUTPUT_DIR, f'predictions_{split}.csv')
    export_predictions(trained_model, export_loader, class_names, save_path)
