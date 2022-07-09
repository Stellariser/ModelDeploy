
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo

"""
semantic: 
    with mix:
        728 in segment head:
        first mIOU: 0.185; second: 0.185
                    Computational complexity:       4.05 GMac (5.81-1.757)
                    Number of parameters:           3.85 M   (4.71-0.839) 

    stem: 
                    Computational complexity:       4.08 GMac (5.84-1.757)
                    Number of parameters:           3.85 M    (4.7-0.853)
                                                    0.186

detail:
        Computational complexity:       11.51 GMac (21.36-9.842)
        Number of parameters:           0.72 M (1.97-1.199)
        mIOU:                           0.13

without booster: 
    Computational complexity:       25.06 GMac -9.842 = 15.21
    Number of parameters:           3.36 M     -1.199 = 
    miou                                                  71 (+5)


with booster:


"""

backbone_url = 'https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth'


class ConvBNReLU(nn.Module): # 3*3conv2d + BatchNorm2d + ReLU
    # in_chan, out_chan
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat


class SepConv(nn.Module): # 3*3conv2d + BatchNorm2d + ReLU
    # in_chan, out_chan
    def __init__(self, in_chan, out_chan):
        super(SepConv, self).__init__()
        # depth-wise convolution
        self.dwconv = nn.Sequential( 
                    nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=1, padding=1, groups=in_chan, bias=False),
                    nn.BatchNorm2d(in_chan),
                    nn.ReLU(inplace=True), 
                )
        # point-wise convolution
        self.pwconv = nn.Sequential(
                    nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(in_chan)
                )

    def forward(self, x):
        feat = self.dwconv(x)
        feat = self.pwconv(feat)
        return feat


class StemBlock(nn.Module):

    def __init__(self, in_chan, out_chan):
        super(StemBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_chan), # 3*3conv 3--->8; 8--->16; 1/2
            nn.Conv2d(out_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan),  # 1*1conv
        )
        self.right = nn.MaxPool2d(  # max 1/2
            kernel_size=3, stride=2, padding=1, ceil_mode=False) 
        mid_chan = in_chan + out_chan # 8+3=11; 16+8=24
        self.fuse = ConvBNReLU(mid_chan, out_chan, 3, stride=1)  # 11-->8; 24-->16

    def forward(self, x):
        feat_left = self.left(x) 
        feat_right = self.right(x) 
        feat = torch.cat([feat_left, feat_right], dim=1) 
        feat = self.fuse(feat) 
        return feat


"""
plus detail 
"""
class DetailBranch(nn.Module):

    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            StemBlock(3, 32), # 1/2
            SepConv(32, 32),
        )
        self.S2 = nn.Sequential(
            StemBlock(32, 64), #1/4
            # ConvBNReLU(64, 64, 3, stride=1),
            SepConv(64, 64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), 
        )
        self.S3 = nn.Sequential(
            StemBlock(64, 128), #1/8
            # ConvBNReLU(128, 128, 3, stride=1),
            SepConv(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128), 
        )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat


"""
semantic block
"""



class ZoomedConv(nn.Module):
    # 只有第二个conv是sepconv: zoomed conv*2 的第二个conv, SS里连续两个3*3conv的第二个conv
    # shortcut, conv, conv+sepconv, zoomed+conv, zoomed+conv; 连接方法： 直接相加

    def __init__(self, in_chan, out_chan, convNum, downsample_factor, upsample_factor):
        super(ZoomedConv, self).__init__()       
        self.downsample_factor = downsample_factor
        if convNum == 1:
            self.convup = nn.Sequential(
                ConvBNReLU(in_chan, out_chan, 3, stride=1), # in--->out   # in can bi equal to out
                nn.Upsample(scale_factor=upsample_factor, mode='bilinear', align_corners=False)
            )  
        elif convNum == 2:
            self.convup = nn.Sequential(
                ConvBNReLU(in_chan, out_chan, 3, stride=1),
                SepConv(in_chan, out_chan),
                nn.Upsample(scale_factor=upsample_factor, mode='bilinear', align_corners=False)
            )  
    def forward(self, x):
        feat = F.interpolate(x, scale_factor= self.downsample_factor, mode='bilinear', align_corners=False)
        feat = self.convup(feat)
        return feat


class SearchSpace(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(SearchSpace, self).__init__()
        self.conv1 = ConvBNReLU(in_chan, out_chan, 3, stride=1)
        self.conv2 = nn.Sequential(
                ConvBNReLU(in_chan, out_chan, 3, stride=1), # in--->out  # in can be equal to out
                SepConv(in_chan, out_chan), # keep out
            ) 
        self.zoomedconv1 = nn.Sequential(
                ZoomedConv(in_chan, out_chan, 1, 0.5, 2), # in--->out  # in can be equal to out
                ConvBNReLU(out_chan, out_chan, 3, stride=1), # keep out
            ) 
        self.zoomedconv2 = nn.Sequential(
                ZoomedConv(in_chan, out_chan, 2, 0.5, 2), # in--->out  # in can be equal to out
                ConvBNReLU(out_chan, out_chan, 3, stride=1), # keep out
            )             
    def forward(self, x):
        feat = x + self.conv1(x) + self.conv2(x) + self.zoomedconv1(x) + self.zoomedconv2(x)
        return feat



class GELayerS2(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, mid_chan, 3, stride=1) # C-->6C
        self.dwconv1 = nn.Sequential(  # 6C
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=2,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.Sequential( # 6C
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential( # 6C --> C
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True

        self.shortcut = nn.Sequential(
                nn.Conv2d( # depth-wise seperable conv
                    in_chan, in_chan, kernel_size=3, stride=2,
                    padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d( # conv
                    in_chan, out_chan, kernel_size=1, stride=1,
                    padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x) # conv
        feat = self.dwconv1(feat) #dwconv s=2,k=6
        feat = self.dwconv2(feat) #dwconv s=1
        feat = self.conv2(feat) # 1*1 conv
        shortcut = self.shortcut(x) # shortcut: dwconv s=2,k=1 + 1*1conv
        feat = feat + shortcut # +shortcut
        feat = self.relu(feat) # relu
        return feat


class CEBlock(nn.Module):
    # input: 16*32  *128
    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0) # 1*1conv: k=1, padding=0
        #TODO: in paper here is naive conv2d, no bn-relu = nn.Conv2d(128, 128, 3， 1)
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

    def forward(self, x): # c一直没有变，=128
        #GAP是对每一个通道的像素求平均，
        #   假如现在的特征图的shape为[B,C,H,W]，经过GAP后shape变为[B,C,1,1]
        feat = torch.mean(x, dim=(2, 3), keepdim=True) # GAP
        feat = self.bn(feat) # BN
        feat = self.conv_gap(feat) # 1*1conv
        feat = feat + x # + shortcut    # 1*1*128 + 16*32*128 每个element相加
        feat = self.conv_last(feat) # conv
        return feat



class FeatureFusion(nn.Module):
    def __init__(self, chan):
        super(FeatureFusion, self).__init__()
        # 1/16: 46; 1/32: 128
        self.up = nn.Sequential(  #128
            ConvBNReLU(chan*2,chan*2, ks=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2),
        )
        self.proj = ConvBNReLU(chan*3, chan*2, 3, stride=1) # 128
        
    def forward(self, x_16, x_32): 
        x_32_up = self.up(x_32)      
        fusion = torch.cat([x_16, x_32_up], dim=1)
        feat = self.proj(fusion)
        return feat



class SegmentBranch(nn.Module):

    def __init__(self):
        super(SegmentBranch, self).__init__()
        self.S1 = StemBlock(3, 8) # stage1,stage2 是stem block # 3-->16, 512*1024-->256*512-->128*256
        self.S2 = StemBlock(8, 16)
        self.S3 = nn.Sequential( # stage3： GE(s=2) + GE
            GELayerS2(16, 32), # 16-->32, 128*256-->64*128
            SearchSpace(32, 32), # 32, 64*128
        )
        self.S4 = nn.Sequential( # stage4： GE(s=2) + GE
            GELayerS2(32, 64), # 32-->64, 64*128-->32*64
            SearchSpace(64, 64), # 64, 32*64
        )
        self.S5_4 = nn.Sequential( # stage5：GE(s=2) + GE * 3 +
            GELayerS2(64, 128), # 64-->128, 32*64-->16*32
            SearchSpace(128, 128), # 128, 16*32
            SearchSpace(128, 128), # 128, 16*32
            SearchSpace(128, 128), # 128, 16*32
        )
        self.S5_5 = CEBlock() # stage5：+ CE # 128, 16*32

        self.F = FeatureFusion(64) # output 128

    def forward(self, x):
        feat1 = self.S1(x)
        feat2 = self.S2(feat1) # feat2: stage1,stage2 的输出
        feat3 = self.S3(feat2)  # feat3: stage3 的输出
        feat4 = self.S4(feat3) # feat4: stage4 的输出
        feat5_4 = self.S5_4(feat4) # feat5_4： stage5 GE后的输出
        feat5_5 = self.S5_5(feat5_4) # feat_S: 加上CE后的输出
        feature_f = self.F(feat4, feat5_5) # 16 + 32
        return feat2, feat3, feat4, feat5_4, feature_f


"""
aggregation
"""
class BGALayer(nn.Module):

    def __init__(self):
        super(BGALayer, self).__init__()

        # left:  1/16, 128
        # right: 1/8, 128
        self.left1 = nn.Sequential( # 不变
            # nn.Conv2d(
            #     128, 128, kernel_size=3, stride=1,
            #     padding=1, groups=128, bias=False),
            # nn.BatchNorm2d(128),
            # nn.Conv2d(
            #     128, 128, kernel_size=1, stride=1,
            #     padding=0, bias=False),
            nn.Identity()
        )
        self.left2 = nn.Sequential( #下采样2
            # nn.Conv2d(
            #     128, 128, kernel_size=3, stride=2,
            #     padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential( #上采样2
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
        )
        self.right2 = nn.Sequential( # 不变
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
            # ZoomedConv(128, 128, 1, 0.5, 2)
            # nn.Identity()
        )
        self.upright = nn.Upsample(scale_factor=2)

        self.convend = nn.Sequential( # 加强
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), # not shown in paper
        )
        
    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
    
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.upright(right) # 最后得到的是1/16,为了和1/8相加，要上采样2
        sum = left + right # 1/8 128
        out = self.convend(sum) # 加强
        return out


"""
train booster
"""

###################### SementHead ###################
class SegmentHead(nn.Module):
    # SegmentHead(128, 1024, n_classes, up_factor=8, aux=False)
    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1) # 3*3conv2d + BatchNorm2d + ReLU
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)
                ) if aux else nn.Identity(),
            nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True), # 1*1conv, k=1,padding=0
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        feat = self.conv(x) # ConvBNReLU: in_chan --> mid_chan
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat


###################### SementHeadPlus ###################
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SegmentHeadPlus(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHeadPlus, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor
        self.se = SELayer(mid_chan, reduction = 16)

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)
                ) if aux else nn.Identity(),
            nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.se(feat)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat



'''

whole net

'''

class BiSeNetPlus(nn.Module): # 继承nn.Module类
    # def __init__(self, n_classes, aux_mode='train')
    def __init__(self, n_classes, aux_mode='train'): 
        super(BiSeNetPlus, self).__init__()

        self.aux_mode = aux_mode 
        self.detail = DetailBranch()  
        self.segment = SegmentBranch() 
        self.bga = BGALayer() 

        if self.aux_mode == 'train': 
            self.aux2 = SegmentHead(16, 128, n_classes, up_factor=4)
            self.aux3 = SegmentHead(32, 128, n_classes, up_factor=8)
            self.aux4 = SegmentHead(64, 128, n_classes, up_factor=16)
            self.aux5_4 = SegmentHead(128, 128, n_classes, up_factor=32)

        self.head = SegmentHead(128, 1024, n_classes, up_factor=8, aux=False)

        # if self.aux_mode == 'train': 
        #     self.aux2 = SegmentHeadPlus(16, 128, n_classes, up_factor=4)
        #     self.aux3 = SegmentHeadPlus(32, 128, n_classes, up_factor=8)
        #     self.aux4 = SegmentHeadPlus(64, 128, n_classes, up_factor=16)
        #     self.aux5_4 = SegmentHeadPlus(128, 128, n_classes, up_factor=32)

        # self.head = SegmentHeadPlus(128, 1024, n_classes, up_factor=8, aux=False)

        self.init_weights() # 初始化权重

    def forward(self, x): # x是输入的图片
        size = x.size()[2:] # 这个变量没有用到
        feat_d = self.detail(x)   # 1/8 128
        feat2, feat3, feat4, feat5_4, feat_f = self.segment(x) # 1/16 128

        feat_head = self.bga(feat_d, feat_f)

        logits = self.head(feat_head) # result feature map (segment head)      
        if self.aux_mode == 'train':
            logits_aux2 = self.aux2(feat2) # result S1,2
            logits_aux3 = self.aux3(feat3) # result S3
            logits_aux4 = self.aux4(feat4) # result S4
            logits_aux5_4 = self.aux5_4(feat5_4) # result S5_4            
            return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4

        if self.aux_mode == 'eval':   #elif
            return logits,  
        elif self.aux_mode == 'pred':
            pred = logits.argmax(dim=1) # PREDICTION
            return pred
        else:
            raise NotImplementedError


    def init_weights(self): # 初始化权重 在__init__末尾
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # self.load_pretrain()


    def load_pretrain(self):  # 加载预训练权重 在def init_weights中
        state = modelzoo.load_url(backbone_url)
        for name, child in self.named_children():
            if name in state.keys():
                child.load_state_dict(state[name], strict=True)


    def get_params(self): # 在train_amp.py的optimizer方法里用到
        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.parameters():
                if param.dim() == 1:
                    nowd_params.append(param)
                elif param.dim() == 4:
                    wd_params.append(param)
                else:
                    print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params





if __name__ == "__main__":
    x = torch.randn(16, 3, 1024, 2048)
    model = BiSeNetPlus(n_classes=19)
    outs = model(x)
    for out in outs:
        print(out.size())

