import torch
import torch.nn as nn

from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss

# 使用 defaultdict 来存储每个 ID 及其对应的索引
from collections import defaultdict
import random
from IPython import embed

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier_vcnet(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class Resnet_Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Resnet_Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        '''
        加载原始模型
        '''
        param_dict = torch.load(trained_path)
        # 去掉state_dict
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']   
        
        # 字典添加
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        # 输出加载信息
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))




# vcnet
from torchvision import models


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier_vcnet)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x


class ft_net(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=2, circle=False):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(2048, class_num, droprate, return_f = circle)

    def forward(self, x,label=None, cam_label= None, view_label=None,isquery=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        feat = x.view(x.size(0), x.size(1))
        if self.training:
            score = self.classifier(feat)
            return score,feat
        else:
            return feat
        
    def load_param(self, trained_path):
        '''
        加载原始模型
        '''
        param_dict = torch.load(trained_path)
        # 去掉state_dict
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']   
        
        # 字典添加
        for i in param_dict:
            try:
                self.state_dict()[i].copy_(param_dict[i])
            except Exception as e:
                embed()
        # 输出加载信息
        print('Loading pretrained model from {}'.format(trained_path))


class view_net(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=2, circle=False):
        super(view_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(2048, class_num, droprate, return_f = circle)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        feat = x.view(x.size(0), x.size(1))
        if self.training:
            
            score = self.classifier(feat)
            return score, feat
        else:
            return feat
    def load_param(self, trained_path):
        '''
        加载原始模型
        '''
        param_dict = torch.load(trained_path)
        # 去掉state_dict
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']   
        
        # 字典添加
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
           
        # 输出加载信息
        print('Loading pretrained model from {}'.format(trained_path))
            

#到此为止后面的模型可以不用看了，是我之前做实验的模型  VCC
# Define the ResNet50-based Model
class ft_net_ours(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=2, circle=False, ibn=False):
        super().__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.circle = circle
        self.proj0 = nn.ConvTranspose2d(in_channels=2, out_channels=3,  kernel_size=16,  stride=16, bias=False)
        self.proj1 = nn.ConvTranspose2d(in_channels=2, out_channels=64, kernel_size=4,   stride=4,  bias=False)
        self.proj2 = nn.ConvTranspose2d(in_channels=2, out_channels=256, kernel_size=4,  stride=4,  bias=False)
        self.proj3 = nn.ConvTranspose2d(in_channels=2, out_channels=512, kernel_size=2,  stride=2,  bias=False)
        self.proj4 = nn.ConvTranspose2d(in_channels=2, out_channels=1024, kernel_size=1, stride=1,  bias=False)

        self.classifier = ClassBlock(2048, class_num, droprate, return_f=circle)

    def forward(self, x, vf): # VCC
        b, c = vf.size()
        # vf = vf.view(b, 2, 16, 16)
        vf = vf.view(b, 2, 1024)
        vf = vf.view(b, 2, 32, 32)
        vf = torch.nn.functional.adaptive_avg_pool2d(vf, (16, 16))
        vf0 = self.proj0(vf)  # b, 3, 256, 256
        x = torch.add(x, vf0)

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x) # b, 64, 64, 64

        vf1 = self.proj1(vf)
        x = torch.add(x, vf1)
        x = self.model.layer1(x)  # b, 256, 64, 64

        vf2 = self.proj2(vf)
        x = torch.add(x, vf2)

        x = self.model.layer2(x)  # b, 512, 32, 32
        vf3 = self.proj3(vf)
        x = torch.add(x, vf3)

        x = self.model.layer3(x)  # b, 1024, 16, 16
        vf4 = self.proj4(vf)
        x = torch.add(x, vf4)

        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        feat = x.view(x.size(0), x.size(1))
        if self.training:
            score = self.classifier(feat)
            return score,feat
        else:
            return feat
    def load_param(self, trained_path):
        '''
        加载原始模型
        '''
        param_dict = torch.load(trained_path)
        # 去掉state_dict
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']   
        
        # 字典添加
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        # 输出加载信息
        print('Loading pretrained model from {}'.format(trained_path))

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768


        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
 
    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)  
        id_to_indices = defaultdict(list)

        # 遍历张量并记录每个 ID 的索引
        for index, value in enumerate(label):
            id_to_indices[value.item()].append(index)
        # print("ID 对应的索引：", dict(id_to_indices))
        ids=id_to_indices.keys()
        for id in ids:
            for i in id_to_indices.get(id):
                print(0)
        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))



def make_model(cfg, num_class, camera_num, view_num,name):
  
    if name == 'vcnet_backbone':
            model= ft_net(class_num=num_class, droprate=0.5, stride=1, circle=False)
            print('===========building vcnet_backbone===========')
    elif name == 'vcnet_view':
            model= view_net(class_num=camera_num, droprate=0.5, stride=1, circle=False)
            print('===========building vcnet_view===========')
    elif name == 'vcnet':
            model= ft_net_ours(class_num=num_class, droprate=0.5, stride=1, circle=False)
            print('===========building ft_net_ours===========')    
    else:
        model = Resnet_Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
