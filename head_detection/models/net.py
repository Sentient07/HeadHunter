import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torch.nn.functional as F
# from .deformconv2d import DeformConv2d
from collections import OrderedDict
from head_detection.models.deformconv2d import DeformConv2d
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool


def conv_bn(inp, oup, filter_size=3, stride=1, pad=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, filter_size, stride, pad, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, filter_size=3, stride=1, pad=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, filter_size, stride, pad, bias=False),
        nn.BatchNorm2d(oup),
    )

def deform_conv_bn(inp, oup, filter_size=3, stride=1, pad=1, leaky=0):
    return nn.Sequential(
        nn.DeformConv2d(inp, oup, filter_size, stride, pad, bias=False, modulation=True),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def deformconv_bn_no_relu(inp, oup, filter_size=3, stride=1, pad=1):
    return nn.Sequential(
        DeformConv2d(inp, oup, filter_size, stride, pad, bias=False, modulation=True),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )


def create_contexts(context, out_channels, n_levels, **kwargs):
    contexts = []
    if context is None:
        return
    if context.lower() == 'cpm':
        context_module = CPM
    elif context.lower() == 'deform_ssh':
        context_module = DeformSSH
    elif context.lower() == 'ssh':
        context_module = SSH
    else:
        raise ValueError("Incorrect context module")
    for _ in range(n_levels):
        contexts.append(context_module(out_channels, out_channels, **kwargs).cuda())
    return contexts


class CPM(nn.Module):
    """Context Module introduced in PyramidBox paper"""

    def __init__(self, in_plane, out_plane=None, **kwargs):
        super(CPM, self).__init__()
        cpmfeat_dim = kwargs.pop('cpmfeat_dim', 1024)
        self.branch1 = conv_bn_no_relu(in_plane, cpmfeat_dim, 1, 1, 0)
        self.branch2a = conv_bn_no_relu(in_plane, 256, 1, 1, 0)
        self.branch2b = conv_bn_no_relu(256, 256, 3, 1, 1)
        self.branch2c = conv_bn_no_relu(256, cpmfeat_dim, 1, 1, 0)

        self.ssh_1 = nn.Conv2d(cpmfeat_dim, 256, kernel_size=3, stride=1, padding=1)
        self.ssh_dimred = nn.Conv2d(
            cpmfeat_dim, 128, kernel_size=3, stride=1, padding=1)
        self.ssh_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.ssh_3a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.ssh_3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.ssh_final = conv_bn_no_relu(512, 256, filter_size=1, stride=1, pad=0)


    def forward(self, inp):
        out_residual = self.branch1(inp)
        x = F.relu(self.branch2a(inp), inplace=True)
        x = F.relu(self.branch2b(x), inplace=True)
        x = self.branch2c(x)
        rescomb = F.relu(x + out_residual, inplace=True)
        ssh1 = self.ssh_1(rescomb)
        ssh_dimred = F.relu(self.ssh_dimred(rescomb), inplace=True)
        ssh_2 = self.ssh_2(ssh_dimred)
        ssh_3a = F.relu(self.ssh_3a(ssh_dimred), inplace=True)
        ssh_3b = self.ssh_3b(ssh_3a)

        ssh_out = torch.cat([ssh1, ssh_2, ssh_3b], dim=1)
        ssh_out = F.relu(self.ssh_final(ssh_out), inplace=True)
        return ssh_out


class SSH(nn.Module):
    """ SSH Context module """
    def __init__(self, in_channel, out_channel, **kwargs):
        super(SSH, self).__init__()
        default_filter = kwargs.pop('default_filter')
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        if default_filter:
            self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2)
            self.conv5X5_1 = conv_bn(in_channel, out_channel//4, leaky = leaky)
            self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4)
            self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
            self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4)
        else:
            self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)
            self.conv5X5_1 = conv_bn(in_channel, out_channel//4, filter_size=5, stride=1, pad=2, leaky=leaky)
            self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, filter_size=5, pad=2, stride=1)
            self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, filter_size=7, stride=1, pad=3, leaky=leaky)
            self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, filter_size=7, pad=3, stride=1)


    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class DeformSSH(nn.Module):
    """ SSH context module with deformable convolution """
    def __init__(self, in_channel, out_channel, **kwargs):
        super(DeformSSH, self).__init__()
        default_filter = kwargs.pop('default_filter')
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = deformconv_bn_no_relu(in_channel, out_channel//2, stride=1)

        if default_filter:
            self.conv5X5_1 = conv_bn(in_channel, out_channel//4, leaky = leaky)
            self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4)
            self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
            self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4)
        else:
            self.conv5X5_1 = conv_bn(in_channel, out_channel//4, filter_size=5, stride=1, pad=2, leaky=leaky)
            self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, filter_size=5, pad=2, stride=1)
            self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, filter_size=7, stride=1, pad=3, leaky=leaky)
            self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, filter_size=7, pad=3, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out


class BackBoneWithFPN(nn.Module):
    """Pytorch's FPN borrowed with deformable conv support"""

    def __init__(self, backbone, return_layers,
                 in_channels_list,
                 out_channels,
                 context_module=None,
                 use_deform=False,
                 default_filter=False):

        self.ssh_list = []
        super(BackBoneWithFPN, self).__init__()
        kwargs = {'default_filter':default_filter}

        self.body = _utils.IntermediateLayerGetter(backbone, return_layers=return_layers)
        # To Support older version
        n_level = len(in_channels_list)
        if n_level == 3:
            self.fpn = FPN(in_channels_list=in_channels_list,
                           out_channels=out_channels)
        elif n_level == 4:
            self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list,
                                             out_channels=out_channels,
                                             extra_blocks=LastLevelMaxPool())
        else:
            raise ValueError("invalid levels")
        self.context_list = create_contexts(context_module,out_channels, n_level,
                                            default_filter=default_filter)
        ### TERRIBLE HACK
        ### Doing this because this is the only way by which we can use pre-trained models
        if self.context_list is not None:
            for ind, context in enumerate(self.context_list):
                cont_str = 'ssh%d'%(ind+1)
                setattr(self, cont_str, context)
        self.out_channels = out_channels

    def forward(self, inputs):
        feat_maps = self.body(inputs)
        fpn_out = self.fpn(feat_maps)
        fpn_out = list(fpn_out.values()) if isinstance(fpn_out, dict) else fpn_out
        if self.context_list is not None:
            context_out = [cont_i(fpn_f) for fpn_f, cont_i in zip(fpn_out, self.context_list)]
        else:
            context_out = fpn_out
        # back to dict
        feat_extracted = OrderedDict([(k, v) for k, v in zip(list(feat_maps.keys()), context_out)])
        return feat_extracted


class MobileNetV1(nn.Module):
    """MobileNet backbone"""
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky = 0.1),    # 3
            conv_dw(8, 16, 1),   # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1), # 59 + 32 = 91
            conv_dw(128, 128, 1), # 91 + 32 = 123
            conv_dw(128, 128, 1), # 123 + 32 = 155
            conv_dw(128, 128, 1), # 155 + 32 = 187
            conv_dw(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2), # 219 +3 2 = 241
            conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x
