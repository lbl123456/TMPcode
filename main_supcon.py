from __future__ import print_function

import os
import sys
import argparse
import time
import math
import os.path as osp
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from glob import glob
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss
from torch.utils.data import Dataset
from data import build
from data.datasets import init_dataset, ImageDataset

import numpy as np
import matplotlib.pyplot as plt
import math
import time
from scipy.spatial.distance import pdist

import torch
from torch.optim.optimizer import Optimizer, required
import re
from PIL import Image
from reid.evaluators import Evaluator
from collections import deque
import random
import os
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.05,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


EETA_DEFAULT = 0.001

class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling for large batch training.
    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0.9,
        use_nesterov=False,
        weight_decay=0.0,
        exclude_from_weight_decay=None,
        exclude_from_layer_adaptation=None,
        classic_momentum=True,
        eeta=EETA_DEFAULT,
    ):
        """Constructs a LARSOptimizer.
        Args:
        lr: A `float` for learning rate.
        momentum: A `float` for momentum.
        use_nesterov: A 'Boolean' for whether to use nesterov momentum.
        weight_decay: A `float` for weight decay.
        exclude_from_weight_decay: A list of `string` for variable screening, if
            any of the string appears in a variable's name, the variable will be
            excluded for computing weight decay. For example, one could specify
            the list like ['batch_normalization', 'bias'] to exclude BN and bias
            from weight decay.
        exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
            for layer adaptation. If it is None, it will be defaulted the same as
            exclude_from_weight_decay.
        classic_momentum: A `boolean` for whether to use classic (or popular)
            momentum. The learning rate is applied during momeuntum update in
            classic momentum, but after momentum for popular momentum.
        eeta: A `float` for scaling of learning rate when computing trust ratio.
        name: The name for the scope.
        """

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            exclude_from_weight_decay=exclude_from_weight_decay,
            exclude_from_layer_adaptation=exclude_from_layer_adaptation,
            classic_momentum=classic_momentum,
            eeta=eeta,
        )

        super(LARS, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eeta = eeta
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
        # arg is None.
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def step(self, epoch=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eeta = group["eeta"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data

                param_state = self.state[p]

                # TODO: get param names
                # if self._use_weight_decay(param_name):
                grad += self.weight_decay * param

                if self.classic_momentum:
                    trust_ratio = 1.0

                    # TODO: get param names
                    # if self._do_layer_adaptation(param_name):
                    w_norm = torch.norm(param)
                    g_norm = torch.norm(grad)

                    device = g_norm.get_device()
                    trust_ratio = torch.where(
                        w_norm.ge(0),
                        torch.where(
                            g_norm.ge(0),
                            (self.eeta * w_norm / g_norm),
                            torch.Tensor([1.0]).to(device),
                        ),
                        torch.Tensor([1.0]).to(device),
                    ).item()

                    scaled_lr = lr * trust_ratio
                    if "momentum_buffer" not in param_state:
                        next_v = param_state["momentum_buffer"] = torch.zeros_like(
                            p.data
                        )
                    else:
                        next_v = param_state["momentum_buffer"]

                    next_v.mul_(momentum).add_(scaled_lr, grad)
                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (scaled_lr * grad)
                    else:
                        update = next_v

                    p.data.add_(-update)
                else:
                    raise NotImplementedError

        return loss

    def _use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True




def load_optimizer(model,batch_size):

    scheduler = None
    # optimized using LARS with linear learning rate scaling
    # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
    learning_rate = 0.3 #* batch_size / 256
    optimizer = LARS(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1.5e-6,
        exclude_from_weight_decay=["batch_normalization", "bias"],
    )

    # "decay the learning rate with the cosine decay schedule without restarts"
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 20, eta_min=0, last_epoch=-1)

    return optimizer, scheduler


class Market(object):

    def __init__(self, root):

        self.images_dir = osp.join(root)
        self.camstyle_path = 'bounding_box_train_camstyle'
        self.camstyle = []
        self.num_camstyle_ids = 0
        self.load()

    def preprocess(self, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(self.images_dir, path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            ret.append((fname, pid, cam))
        return ret, int(len(all_pids))

    def load(self):
        self.camstyle, self.num_camstyle_ids = self.preprocess(self.camstyle_path)
        print("  camstyle  | {:5d} | {:8d}"
              .format(self.num_camstyle_ids, len(self.camstyle)))

class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid
class RandomErasing(object):
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2

    def __call__(self, img):
        
        if np.random.rand() > self.p:
            return img
        
        img = np.array(img)
        
        while True:
            img_h, img_w, img_c = img.shape

            img_area = img_h * img_w
            mask_area = np.random.uniform(self.sl, self.sh) * img_area
            mask_aspect_ratio = np.random.uniform(self.r1, self.r2)
            mask_w = int(np.sqrt(mask_area / mask_aspect_ratio))
            mask_h = int(np.sqrt(mask_area * mask_aspect_ratio))
            
            mask = np.random.rand(mask_h, mask_w, img_c) * 255

            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)
            right = left + mask_w
            bottom = top + mask_h
        
            if right <= img_w and bottom <= img_h:
                break
        
        img[top:bottom, left:right, :] = mask
        
        return Image.fromarray(img)

class RandomPatch(object):
    """Random patch data augmentation.
    输入是 ： hwc 0-255
    和 随机擦除是一致差不多的， 都是像素块遮挡，区别在于，这个遮挡区域不是灰色块，是 图片btach ,随机的一个面积放进去的
    There is a patch pool that stores randomly extracted pathces from person images.
    For each input image, RandomPatch
        1) extracts a random patch and stores the patch in the patch pool;
        2) randomly selects a patch from the patch pool and pastes it on the
           input (at random position) to simulate occlusion.
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.

          min_sample_size   和 batch 有关系
          batch 64  min_sample_size=60  61张图片原来的样子， 3张处理后的图片

    """

    def __init__(self, prob_happen=1, pool_capacity=50000, min_sample_size=5,
                 patch_min_area=0.01, patch_max_area=0.5, patch_min_ratio=0.1,
                 prob_rotate=0.5, prob_flip_leftright=0.5,
                 ):

        self.prob_happen = prob_happen

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio

        self.prob_rotate = prob_rotate
        self.prob_flip_leftright = prob_flip_leftright

        self.patchpool = deque(maxlen=pool_capacity)

        self.min_sample_size = min_sample_size

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(self.patch_min_area, self.patch_max_area) * area
            aspect_ratio = random.uniform(self.patch_min_ratio, 1. / self.patch_min_ratio)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h
        return None, None

    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.uniform(0, 1) > self.prob_rotate:
            patch = patch.rotate(random.randint(-10, 10))
        return patch

    def __call__(self, img):
        W, H = img.size  # original image size

        # collect new patch
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)            
            new_patch = img.crop((x1, y1, x1 + w, y1 + h))  #剪切一部分图片


            self.patchpool.append(new_patch)
        #print("**************************")
        if len(self.patchpool) < self.min_sample_size:
            #print(len(self.patchpool))
            # print(np.self.patchpool)
            #print(self.min_sample_size)
            return img

        if random.uniform(0, 1) > self.prob_happen:
            return img

        # paste a randomly selected patch on a random position
        patch = random.sample(self.patchpool, 1)[0]
        patchW, patchH = patch.size
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)
        img.paste(patch, (x1, y1))

        return img

def set_loader(opt):
    # construct data loader
    '''if opt.dataset == 'cifar10':
        mean = (0.485,0.456,0.406)
        std = (0.229,0.224,0.225)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.mean)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))'''

    #加载camstyle数据集
    #root = 'C:\\Users\\DELL\\Desktop\\SupContrast-master\\data\\market1501'
    #CamStyle_dataset = Market(root)


    mean = (0.485,0.456,0.406)
    std = (0.229,0.224,0.225)
    size = (256,128)
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        #transforms.Resize(size=(256,256)),                              #先调整至fakeimg 的size，(统一尺寸)方便进行RandomPatch
        #RandomPatch(),                                                 #随机补丁
        transforms.RandomResizedCrop(size=size),                        #随机裁剪
        transforms.RandomHorizontalFlip(),                              #随机水平翻转                 
        transforms.RandomRotation(180),                                 #随机旋转
        #transforms.Resize(size=size),                                   #resize
        transforms.RandomGrayscale(p=0.2),                              #将图像以一定的概率转换为灰度图像
        RandomErasing(),                                                #随机擦除
        transforms.ToTensor(),
        normalize,  
    ])
    source_transform = transforms.Compose([
        transforms.Resize(size=size),
        transforms.ToTensor(),
        normalize,
    ])
    NAMES = 'market1501'
    DIR = os.getcwd()
    ROOT_DIR = DIR+'\\data'
    dataset = init_dataset(NAMES, root=ROOT_DIR)
    #加载Camstyle图像在ImageDataset中
    train_dataset = ImageDataset(dataset.train,TwoCropTransform(train_transform,source_transform))
    '''if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)'''

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)


    query_loader = torch.utils.data.DataLoader(
        Preprocessor(dataset.query,
                     root=osp.join(dataset.dataset_dir, dataset.query_dir), transform=source_transform),
        batch_size=opt.batch_size, num_workers=opt.num_workers,
        shuffle=False, pin_memory=True)

    gallery_loader = torch.utils.data.DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.dataset_dir, dataset.gallery_dir), transform=source_transform),
        batch_size=opt.batch_size, num_workers=opt.num_workers,
        shuffle=False, pin_memory=True)

    return train_loader,query_loader,gallery_loader,dataset


def set_model(opt):
    #model
    model = SupConResNet(name=opt.model)
    #loss
    criterion = SupConLoss(temperature=opt.temp)


    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    #序号；images三张，数据增强两张，原图一张；真实标签（未使用）；；   
    for idx,(images, labels,camera_id,image_path) in enumerate(train_loader):
        data_time.update(time.time() - end)

        #数据增强 2*bs 张
        images_1 = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images_1 = images_1.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        if epoch <= 2:
            warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images_1)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)


        #原始图像，聚类算法生成label
        real_image = images[2]
        if torch.cuda.is_available():
            real_image = real_image.cuda(non_blocking=True)
        features_realimage = model(real_image)
        features_realimage = features_realimage.cpu()
        features_realimage = features_realimage.detach().numpy()
        features_realimage = np.mat(features_realimage).transpose()
        clusters, clusterNum = dbscan(features_realimage, 0.75, 1)
        labels = torch.Tensor(clusters)


        #测试loss
        '''features = torch.Tensor([[[1,2],[4,3]],[[1,1],[2,2]]])
        features.cuda()
        labels = torch.Tensor([1,2])
        labels.cuda()'''

        if clusterNum != bsz:
            print(clusterNum)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg

UNCLASSIFIED = False
NOISE = 0

def dist(a, b):
    #v1 = math.sqrt(np.power(a - b, 2).sum())
    #return v1
    up=np.double(np.bitwise_and((a != b),np.bitwise_or(a != 0, b != 0)).sum())
    down=np.double(np.bitwise_or(a != 0, b != 0).sum())
    d1=(up/down)
    return d1


    #X=np.vstack([a,b])
    #d2=pdist(X,'jaccard')  # 算出来的就是jaccard距离，需要计算jaccard系数的话就需要1-d2
    #return d2
	
def eps_neighbor(a, b, eps):
	return dist(a, b) < eps

def region_query(data, pointId, eps):
	nPoints = data.shape[1]
	seeds = []
	for i in range(nPoints):
		if eps_neighbor(data[:, pointId], data[:, i], eps):
			seeds.append(i)
	return seeds

def expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
	seeds = region_query(data, pointId, eps)
	if len(seeds) < minPts: # 不满足minPts条件的为噪声点
		clusterResult[pointId] = NOISE
		return False
	else:
		clusterResult[pointId] = clusterId # 划分到该簇
		for seedId in seeds:
			clusterResult[seedId] = clusterId

		while len(seeds) > 0: # 持续扩张
			currentPoint = seeds[0]
			queryResults = region_query(data, currentPoint, eps)
			if len(queryResults) >= minPts:
				for i in range(len(queryResults)):
					resultPoint = queryResults[i]
					if clusterResult[resultPoint] == UNCLASSIFIED:
						seeds.append(resultPoint)
						clusterResult[resultPoint] = clusterId
					elif clusterResult[resultPoint] == NOISE:
						clusterResult[resultPoint] = clusterId
			seeds = seeds[1:]
		return True

def dbscan(data, eps, minPts):
	clusterId = 1
	nPoints = data.shape[1]
	clusterResult = [UNCLASSIFIED] * nPoints
	for pointId in range(nPoints):
		point=data[:, pointId]  
		if clusterResult[pointId] == UNCLASSIFIED:
			if expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
				clusterId = clusterId + 1
	return clusterResult, clusterId - 1

def main():
    #初始化配置
    opt = parse_option()

    #加载数据集
    train_loader,query_loader,gallery_loader,dataset = set_loader(opt)

    #构建模型，loss函数，  ResNet50  加载imagenet预训练权重
    model, criterion = set_model(opt)

    #构建优化器
    #optimizer = set_optimizer(opt, model)
    optimizer, scheduler = load_optimizer(model,opt.batch_size)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    #evaluator = Evaluator(model)
    #evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, 2048, True)
    

    # training routine
    for epoch in range(1, opt.epochs + 1):
        #adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()

        #对每一个eproch聚类 费时长
        '''for idx in train_loader:
            labels = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'''
        '''for inx in train_loader.dataset:
            inx[1] = 0'''
        '''for idx,(images, labels,camera_id,image_path) in enumerate(train_loader):
            labels = torch.Tensor([int(0),int(0),int(0),int(0),int(0),int(0),int(0),int(0)])
            print(idx)'''
        '''features = []
        for (image1,image2,image),label,camera_id,path in train_loader.dataset:
            v1 = torch.unsqueeze(image,0)
            v1 = v1.cuda(non_blocking=True)
            feature = model(v1)
            feature = feature.cpu()
            feature = feature.detach().numpy()
            feature = feature[0]
            features.append(feature)
            #if len(features) == 100:
                #break
            #print(len(features))
        features = np.mat(features).transpose()
        clusters, clusterNum = dbscan(features, 0.75, 1)'''
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        #更新学习率
        scheduler.step()
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)


        if epoch == 1 or epoch % 10 == 0:
            evaluator = Evaluator(model)
            with torch.no_grad():
                evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, 2048, True)


        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
