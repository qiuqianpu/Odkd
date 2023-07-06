import torch
import torch.nn as nn

from nets.ConvNext import ConvNeXt_Small, ConvNeXt_Tiny
from nets.CSPdarknet import C3, Conv, CSPDarknet
from nets.Swin_transformer import Swin_transformer_Tiny
from pygcn.models import GCN

import numpy as np

# SE
# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)


#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, adj, backbone='cspdarknet', pretrained=False, input_shape=[640, 640]):
        super(YoloBody, self).__init__()
        depth_dict          = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        #-----------------------------------------------#
        self.backbone_name  = backbone
        self.num_classes = num_classes
        self.adj = adj

        if backbone == "cspdarknet":
            #---------------------------------------------------#   
            #   生成CSPdarknet53的主干模型
            #   获得三个有效特征层，他们的shape分别是：
            #   80,80,256
            #   40,40,512
            #   20,20,1024
            #---------------------------------------------------#
            self.backbone   = CSPDarknet(base_channels, base_depth, phi, pretrained)
        else:
            #---------------------------------------------------#   
            #   如果输入不为cspdarknet，则调整通道数
            #   使其符合YoloV5的格式
            #---------------------------------------------------#
            self.backbone       = {
                'convnext_tiny'         : ConvNeXt_Tiny,
                'convnext_small'        : ConvNeXt_Small,
                'swin_transfomer_tiny'  : Swin_transformer_Tiny,
            }[backbone](pretrained=pretrained, input_shape=input_shape)
            in_channels         = {
                'convnext_tiny'         : [192, 384, 768],
                'convnext_small'        : [192, 384, 768],
                'swin_transfomer_tiny'  : [192, 384, 768],
            }[backbone]
            feat1_c, feat2_c, feat3_c = in_channels 
            self.conv_1x1_feat1 = Conv(feat1_c, base_channels * 4, 1, 1)
            self.conv_1x1_feat2 = Conv(feat2_c, base_channels * 8, 1, 1)
            self.conv_1x1_feat3 = Conv(feat3_c, base_channels * 16, 1, 1)
            
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3         = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1    = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.conv_for_feat2         = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2    = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1  = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2  = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        # 80, 80, 256 => 80, 80, 3 * (5 + num_classes) => 80, 80, 3 * (4 + 1 + num_classes)
        # self.yolo_head_P3_box = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.yolo_head_P3_box = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * 4, 1)
        self.yolo_head_P3_cls = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (1 + num_classes), 1)
        # 40, 40, 512 => 40, 40, 3 * (5 + num_classes) => 40, 40, 3 * (4 + 1 + num_classes)
        #self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_P4_box = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * 4, 1)
        self.yolo_head_P4_cls = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (1 + num_classes), 1)
        # 20, 20, 1024 => 20, 20, 3 * (5 + num_classes) => 20, 20, 3 * (4 + 1 + num_classes)
        #self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)
        self.yolo_head_P5_box = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * 4, 1)
        self.yolo_head_P5_cls = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (1 + num_classes), 1)

        self.GCN = GCN(nfeat=1,
                    nhid=16,
                    nclass=1,
                    dropout=0.5)

        # self.maxpool2 = nn.MaxPool2d(52)
        # self.maxpool1 = nn.MaxPool2d(26)
        # self.maxpool0 = nn.MaxPool2d(13)
        #
        # self.maxpool3 = nn.MaxPool1d(9)


    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone(x)
        if self.backbone_name != "cspdarknet":
            feat1 = self.conv_1x1_feat1(feat1)
            feat2 = self.conv_1x1_feat2(feat2)
            feat3 = self.conv_1x1_feat3(feat3)

        # 20, 20, 1024 -> 20, 20, 512
        P5          = self.conv_for_feat3(feat3)
        # 20, 20, 512 -> 40, 40, 512
        P5_upsample = self.upsample(P5)
        # 40, 40, 512 -> 40, 40, 1024
        P4          = torch.cat([P5_upsample, feat2], 1)

        # 40, 40, 1024 -> 40, 40, 512
        P4          = self.conv3_for_upsample1(P4)

        # 40, 40, 512 -> 40, 40, 256
        P4          = self.conv_for_feat2(P4)
        # 40, 40, 256 -> 80, 80, 256
        P4_upsample = self.upsample(P4)
        # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        P3          = torch.cat([P4_upsample, feat1], 1)

        # 80, 80, 512 -> 80, 80, 256
        P3          = self.conv3_for_upsample2(P3)
        
        # 80, 80, 256 -> 40, 40, 256
        P3_downsample = self.down_sample1(P3)
        # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
        P4 = torch.cat([P3_downsample, P4], 1)

        # 40, 40, 512 -> 40, 40, 512
        P4 = self.conv3_for_downsample1(P4)

        # 40, 40, 512 -> 20, 20, 512
        P4_downsample = self.down_sample2(P4)
        # 20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
        P5 = torch.cat([P4_downsample, P5], 1)

        # 20, 20, 1024 -> 20, 20, 1024
        P5 = self.conv3_for_downsample2(P5)

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,80,80)
        #---------------------------------------------------#
        #out2 = self.yolo_head_P3(P3)

        out2_box = self.yolo_head_P3_box(P3)
        out2_cls = self.yolo_head_P3_cls(P3)
        bs = out2_cls.size(0)
        #print("2:", out2_box.shape)
        #print("2:", out2_cls.shape)
        # 拼接后计算损失
        t2_box = out2_box.view(bs, 3, 4, 52 * 52)
        t2_cls = out2_cls.view(bs, 3, (1 + self.num_classes), 52 * 52)
        out2 = torch.cat([t2_box, t2_cls], dim=2)
        out2 = out2.view([bs, -1, 52, 52])


        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,40,40)
        #---------------------------------------------------#
        # out1 = self.yolo_head_P4(P4)
        out1_box = self.yolo_head_P4_box(P4)
        out1_cls = self.yolo_head_P4_cls(P4)
        # 拼接后计算损失
        t1_box = out1_box.view(bs, 3, 4, 26 * 26)
        t1_cls = out1_cls.view(bs, 3, (1 + self.num_classes), 26 * 26)
        out1 = torch.cat([t1_box, t1_cls], dim=2)
        out1 = out1.view([bs, -1, 26, 26])


        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,20,20)
        #---------------------------------------------------#
        # out0 = self.yolo_head_P5(P5)

        out0_box = self.yolo_head_P5_box(P5)
        out0_cls = self.yolo_head_P5_cls(P5)
        # 拼接后计算损失
        t0_box = out0_box.view(bs, 3, 4, 13 * 13)
        t0_cls = out0_cls.view(bs, 3, (1 + self.num_classes), 13 * 13)
        out0 = torch.cat([t0_box, t0_cls], dim=2)
        out0 = out0.view([bs, -1, 13, 13])




        # 以下语义关联模块

        # 一、原始概率矩阵

        # -----------------------------------------------#
        #   原始类别头输出一共三个，他们的shape分别是
        #   bs, 3 * (1+num_classes), 52, 52 => bs, 3, 1 + num_classes, 52, 52 => batch_size, 3, 52, 52, 1 + num_classes

        #   batch_size, 3, 13, 13, 1 + num_classes
        #   batch_size, 3, 26, 26, 1 + num_classes
        #   batch_size, 3, 52, 52, 1 + num_classes
        tensor2_org = out2_cls.view(bs, 3, (1 + self.num_classes), 52, 52).permute(0, 1, 3, 4, 2).contiguous()
        tensor1_org = out1_cls.view(bs, 3, (1 + self.num_classes), 26, 26).permute(0, 1, 3, 4, 2).contiguous()
        tensor0_org = out0_cls.view(bs, 3, (1 + self.num_classes), 13, 13).permute(0, 1, 3, 4, 2).contiguous()

        # 取(1 + self.num_classes)中类别概率并排为矩阵形式
        matrix2_org = tensor2_org.view(-1, 3 * 52 * 52, (1 + self.num_classes))
        matrix1_org = tensor1_org.view(-1, 3 * 26 * 26, (1 + self.num_classes))
        matrix0_org = tensor0_org.view(-1, 3 * 13 * 13, (1 + self.num_classes))

        # 原始置信+概率矩阵 [bs, num_anchors, (1 + num_classes)]
        GCN_input_org = torch.cat([matrix2_org, matrix1_org, matrix0_org], dim=1)
        Matrix_org = torch.sigmoid(GCN_input_org)

        # 置信系数{0,1} [bs, num_anchors, 1]
        matrix_conf_org = Matrix_org[..., 0].unsqueeze(dim=2)
        matrix_conf = torch.where(matrix_conf_org > 0.5, 1, 0)

        # 置信系数{0,1}取舍后的概率矩阵[bs, num_anchors, num_classes]
        matrix_cls_org = Matrix_org[..., 1:]
        matrix_cls = torch.mul(matrix_conf, matrix_cls_org).detach()

        # 取每个类别的最大概率值为GCN_input
        GCN_input = torch.max(matrix_cls, dim=1)[0].unsqueeze(dim=2).detach()
        # GCN_input_ = torch.where(GCN_input > 0.3, 1.0, 0.0)
        # GCN_input_ = torch.cat([GCN_input, GCN_input_], dim=2)
        #print("1", GCN_input_)


        # meta_target = [[1.0], [3.0], [1.0], [1.0], [1.0], [3.0], [1.0], [3.0]]
        # meta_target = np.array(meta_target)
        # meta_target = torch.from_numpy(meta_target)
        # meta_target = meta_target.cuda()
        # meta_target = meta_target.to(torch.float32)

        # meta_target = meta_target - GCN_input_

        # GCN处理
        # 邻接矩阵，转换为tensor后需指定为torch.float32

        # print(GCN_input.shape[2])
        # nfeat取结点特征维度， nhid为中间层维度， nclass为输出维度
        # model = GCN(nfeat=GCN_input.shape[2],
        #             nhid=16,
        #             nclass=self.num_classes,
        #             dropout=0.5)

        output = self.GCN(GCN_input, self.adj)
        # print(output)
        # print("2", meta)
        meta_list = []
        meta_list.append(GCN_input)
        meta_list.append(output)


        # # 对角矩阵化
        # output = output.squeeze(axis=2)
        # output = torch.diag_embed(output)
        # #print(output)
        #
        # # 调整后的置信+概率矩阵 [bs, num_anchors, (1 + num_classes)]
        # matrix_cls_fixed = torch.bmm(Matrix_org[..., 1:], output)
        # Matrix_fixed = torch.cat([matrix_conf_org, matrix_cls_fixed], dim=2)
        # # print(matrix_conf_org == Matrix_fixed[..., 0].unsqueeze(dim=2))

        # 加法调整
        # 调整后的置信+概率矩阵 [bs, num_anchors, (1 + num_classes)]
        matrix_cls_fixed = torch.sigmoid(GCN_input_org[..., 1:] + output.permute(0, 2, 1))
        Matrix_fixed = torch.cat([matrix_conf_org, matrix_cls_fixed], dim=2)
        # print(matrix_conf_org == Matrix_fixed[..., 0].unsqueeze(dim=2))


        tensor_ = Matrix_fixed.permute(0, 2, 1)
        tensor2 = tensor_[:, :, 0:8112].permute(0, 2, 1)
        tensor1 = tensor_[:, :, 8112:(8112 + 2028)].permute(0, 2, 1)
        tensor0 = tensor_[:, :, (8112 + 2028):10647].permute(0, 2, 1)


        # 经过图卷积调整后的分类头
        cls2_ = tensor2.view(bs, 3, 52, 52, (1 + self.num_classes)).permute(0, 1, 4, 2, 3)
        cls1_ = tensor1.view(bs, 3, 26, 26, (1 + self.num_classes)).permute(0, 1, 4, 2, 3)
        cls0_ = tensor0.view(bs, 3, 13, 13, (1 + self.num_classes)).permute(0, 1, 4, 2, 3)


        t2_cls_ = cls2_.view(bs, 3, (1 + self.num_classes), 52 * 52)
        out2_fix = torch.cat([t2_box, t2_cls_], dim=2)
        out2_fix = out2_fix.view([bs, -1, 52, 52])

        t1_cls_ = cls1_.view(bs, 3, (1 + self.num_classes), 26 * 26)
        out1_fix = torch.cat([t1_box, t1_cls_], dim=2)
        out1_fix = out1_fix.view([bs, -1, 26, 26])

        t0_cls_ = cls0_.view(bs, 3, (1 + self.num_classes), 13 * 13)
        out0_fix = torch.cat([t0_box, t0_cls_], dim=2)
        out0_fix = out0_fix.view([bs, -1, 13, 13])

        # return out0, out1, out2
        return out0, out1, out2, out0_fix, out1_fix, out2_fix, meta_list
        # return out0, out1, out2, out0, out1, out2, meta

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.rand([8, 3, 416, 416]).to(device)

    # 邻接矩阵，转换为tensor后需指定为torch.float32
    adj = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    # torch.float32
    adj = torch.tensor(adj, dtype=torch.float).to(device)

    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes = 3
    phi = 's'
    model = YoloBody(anchors_mask, num_classes, phi, adj, input_shape=[416, 416])
    model.to(device)
    out0, out1, out2, out0_fix, out1_fix, out2_fix, meta = model(input)
    # print(out0.shape)
    # print(out1.shape)
    # print(out2.shape)
    # print(out0_fix.shape)
    # print(out1_fix.shape)
    # print(out2_fix.shape)

    #print('meta:', meta[0])
    print('meta_:', meta[1])
    # outputs = model(input)[-3:]
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    loss_ = torch.nn.MSELoss()

    print(loss_(meta[0], meta[1]))