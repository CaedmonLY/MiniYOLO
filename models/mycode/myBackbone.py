"""

直接从 torchvision.models 里面拷贝的，用于学习

"""

import torch
import torch.nn as nn
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url

__all__ = [
    'ShuffleNetV2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
]

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class downSample(nn.Module):
    def __init__(self, inp, oup, stride):
        super(downSample, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out



class cba(nn.Module):
    def __init__(self,i, o, kernel_size, stride=1, padding=0, bias=False):
        super(cba, self).__init__()
        self.conv=nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)
        self.bn=nn.BatchNorm2d(o)
        self.active=nn.ReLU(inplace=True)

    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.active(x)
        return x




# 3*3卷积之间不用加激活函数？
class mutilspecs(nn.Module):
    def __init__(self, inp,oup):
        super(mutilspecs, self).__init__()

        if not (inp%4==0):
            raise ValueError('mutilspecs 的输入特征通道数必须是4的整数倍')

        innerChannle=inp

        self.sc0=cba(inp,innerChannle,3,1,1)                    # x
        self.sc1=cba(innerChannle//2,innerChannle//2,3,1,1)     # 0.5x
        self.sc2=cba(innerChannle//4,innerChannle//4,3,1,1)     # 0.25x
        #self.mix=nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False,groups=2)
        self.mix12=nn.Conv2d(oup, oup, kernel_size=1, stride=1, padding=0, bias=False,groups=2)
        self.mix0=nn.Conv2d(innerChannle//2, innerChannle//2, kernel_size=1, stride=1, padding=0, bias=False,groups=1)

    def forward(self, x):

        x0=self.sc0(x)
        x0,x=x0.chunk(2, dim=1)         # x0   0.5x

        x1=self.sc1(x)
        x1,x=x1.chunk(2,dim=1)          # x1   0.25x

        x2=self.sc2(x)                  # x2   0.25x
        x0=self.mix0(x0)
        x12=torch.cat((x1,x2),dim=1)



        out=torch.cat((x0,x12),dim=1)
        out = self.mix12(out)


        return out


def mutilspecsTest():
    data=torch.randn(1,196,20,20)
    net=mutilspecs(196,196)
    #m(data)
    #torch.save(m.state_dict(),"mutilspecs")
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(net, (196, 10, 10), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


    net(data)






from models.mycode.fpn import FPN

class DWConvBlock():
    def __init__(self,inputChannel,repeat=1):
        super(convBlock, self).__init__()

        #self.convList=nn.ModuleList()
        self.convList=nn.Sequential()
        for i in range(repeat):
            self.convList.append(cba(inputChannel,
                                     inputChannel,3,1,1,bias=True))

    def forward(self,x):

        return self.convList(x)
        pass


"""
class resDWConv(nn.Module):
    def __init__(self, inp, oup):
        super(resDWConv, self).__init__()
        branch_features = oup // 2
        self.branch0= nn.Sequential(
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )
        self.branch1= nn.Sequential(
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

        self.mix=nn.Sequential(
            nn.Conv2d(oup+oup, oup, kernel_size=1, stride=1, padding=0, bias=False,groups=2),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
        self.branch2= nn.Sequential(
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        x0=x
        x1, x2 = x.chunk(2, dim=1)
        print(x1.shape,x2.shape)
        x = torch.cat((x1, self.branch0(x2)), dim=1)
        tmp1 = channel_shuffle(x, 2)

        x1,x2=tmp1.chunk(2, dim=1)
        x = torch.cat((x1, self.branch1(x2)), dim=1)
        tmp2 = channel_shuffle(x, 2)

        x=torch.cat([tmp1,tmp2],dim=1)
        print(x.shape)
        x=self.mix(x)

        x1,x2=x.chunk(2, dim=1)
        x = torch.cat((x1, self.branch2(x2)), dim=1)
        out = channel_shuffle(x, 2)

        return out
"""

class resDWConv(nn.Module):
    def __init__(self, inp, oup):
        super(resDWConv, self).__init__()

        if not (oup%4==0):
            raise ValueError('resDWConv 的输出特征通道数必须是4的整数倍')

        branch_features = oup // 2
        self.branch0= nn.Sequential(
            # nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(branch_features),
            # nn.ReLU(inplace=True),
            # self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(branch_features),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )
        self.branch1= nn.Sequential(
            # nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(branch_features),
            # nn.ReLU(inplace=True),
            # self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(branch_features),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

        self.mix1=nn.Sequential(
            nn.Conv2d(oup,int(0.25*oup), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(0.25*oup)),
            nn.ReLU(inplace=True),
        )

        self.mix2=nn.Sequential(
            nn.Conv2d(oup,int(0.75*oup), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(0.75*oup)),
            nn.ReLU(inplace=True),
        )

        self.branch2= nn.Sequential(
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x = torch.cat((x1, self.branch0(x2)), dim=1)
        tmp1 = channel_shuffle(x, 2)

        x1,x2=tmp1.chunk(2, dim=1)
        x = torch.cat((x1, self.branch1(x2)), dim=1)
        tmp2 = channel_shuffle(x, 2)


        tmp1=self.mix1(tmp1)
        tmp2=self.mix2(tmp2)
        x=torch.cat([tmp1,tmp2],dim=1)

        x1,x2=x.chunk(2, dim=1)
        x = torch.cat((x1, self.branch2(x2)), dim=1)
        out = channel_shuffle(x, 2)

        return out


class DWPAN(nn.Module):
    def __init__(self,
                 inChannels=[116,116,116],
                 outChannels=116,
                 numOuts=3):
        super(DWPAN, self).__init__()
        self.fpn=FPN(inChannels,outChannels,numOuts)
        """
        self.det0=DWConvBlock(116)
        self.det1=DWConvBlock(116)
        self.det2=DWConvBlock(116)
        """
        self.det0=resDWConv(outChannels,outChannels)

        self.det1=resDWConv(outChannels,outChannels)

        self.det2=resDWConv(outChannels,outChannels)

        self.c01=nn.Sequential(
            self.depthwise_conv(outChannels, outChannels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(outChannels),
        )
        self.l01=nn.Sequential(
            nn.Conv2d(int(2*outChannels), outChannels, kernel_size=1, stride=1, padding=0, bias=False, groups=2),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
        )
        self.c12=nn.Sequential(
            self.depthwise_conv(outChannels, outChannels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(outChannels),
        )
        self.l12=nn.Sequential(
            nn.Conv2d(int(2*outChannels), outChannels, kernel_size=1, stride=1, padding=0, bias=False, groups=2),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)


    def forward(self,x):
        x=self.fpn(x)
        det0=self.det0(x[0])


        t=self.c01(det0)
        t=torch.cat((x[1],t),dim=1)
        t = self.l01(t)
        det1=self.det1(t)

        t=self.c12(det1)
        t=torch.cat((x[2],t),dim=1)
        t = self.l12(t)
        det2=self.det2(t)


        return [det0,det1,det2]


from models.mycode.pan import PAN

from models.mycode.backbone import C3s


class mainResBloacBackbone(nn.Module):
    def __init__(self,stage_out_channels=[24,116,232,464],stages_repeats=[1,2,1],detChannel=196):
    #def __init__(self, stage_out_channels=[24, 96,192, 384], stages_repeats=[1, 2, 1]):
    #def __init__(self, stage_out_channels=[24, 196, 196, 196], stages_repeats=[1, 2, 1]):
        super(mainResBloacBackbone, self).__init__()

        output_channels =stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, output_channels, 3, 2, 1, bias=False,groups=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stages=nn.ModuleList()
        input_channels = output_channels
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, stage_out_channels[1:]):

            seq = [InvertedResidual(input_channels, output_channels, 2)]



            for i in range(repeats):
                seq.append(resDWConv(output_channels,output_channels))
            seq.append(mutilspecs(output_channels,output_channels))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        #self.pan=DWPAN([stage_out_channels[1],stage_out_channels[2],stage_out_channels[3]],detChannel)

        self.pan= PAN([stage_out_channels[1], stage_out_channels[2], stage_out_channels[3]], 116,3)

        self.c3=C3s()

        self.pan2=PAN([116,116,116],116,3)

        self.c3b=C3s()

        pass

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)              # 38*38
        out.append(x)

        x = self.stage3(x)              # 19*19
        out.append(x)

        x = self.stage4(x)              # 10*10
        out.append(x)

        #out=self.pan(out)

        add=out

        # mybackbone+cascadePan
        out=self.pan(out)
        out=self.c3(out)
        out=self.pan2(out)
        out=self.c3b(out)

        return out



def testMainResBlockBackbone():
    net=mainResBloacBackbone()
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(net, (3, 320, 320), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))





class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000):
        super(ShuffleNetV2, self).__init__()


        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False,groups=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        #detChannel=stages_out_channels[1]
        self.pan=DWPAN([stages_out_channels[1],stages_out_channels[2],stages_out_channels[3]],196)
        #self.pan=PAN([stages_out_channels[1],stages_out_channels[2],stages_out_channels[3]],116)

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)              # 38*38
        out.append(x)

        x = self.stage3(x)              # 19*19
        out.append(x)

        x = self.stage4(x)              # 10*10
        out.append(x)

        out=self.pan(out)

        return out




class ShuffleNetV2_org(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000):
        super(ShuffleNetV2_org, self).__init__()


        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False,groups=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels


    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)              # 38*38
        out.append(x)

        x = self.stage3(x)              # 19*19
        out.append(x)

        x = self.stage4(x)              # 10*10
        out.append(x)


        return out


def _shufflenetv2(arch, pretrained, progress, *args, **kwargs):
    model = ShuffleNetV2(*args, **kwargs)

    if pretrained:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)

    return model





def shufflenet_v2_x0_5(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x0.5', pretrained, progress,
                         [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)


def shufflenet_v2_x1_0(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    # 760kb -> 538kb
    """
    return _shufflenetv2('shufflenetv2_x1.0', pretrained, progress,
                         [4, 8, 4], [24, 196, 196, 196, 1024], **kwargs)
    """

    return _shufflenetv2('shufflenetv2_x1.0', pretrained, progress,
                         [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)
                         #              116, 232, 464   这三个是可用的通道


def shufflenet_v2_x1_5(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.5', pretrained, progress,
                         [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)


def shufflenet_v2_x2_0(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 2.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x2.0', pretrained, progress,
                         [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)



def textNet():
    img416=torch.randn(1,3,416,416).float()
    img300 = torch.randn(1, 3, 300, 300).float()
    net=shufflenet_v2_x1_0()

    out=net(img416)

    #print(net)
    for i in out:
        print(i.shape)
    #out=net(img300)




    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(net, (3, 416, 416), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))



def netTest():

    net=shufflenet_v2_x1_0()
    #net=InvertedResidual(116,116,1)

    #m(data)
    #torch.save(m.state_dict(),"mutilspecs")
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(net, (3, 320, 320), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    pass

"""
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPS

"""

class MIX512(nn.Module):
    def __init__(self,ic=512):
        super(MIX512, self).__init__()
        self.f0=nn.ModuleList()
        self.f1 = nn.ModuleList()
        self.f2 = nn.ModuleList()
        self.f3 = nn.ModuleList()
        self.f4 = nn.ModuleList()
        self.fc=nn.Conv2d(512,96,1,groups=2)


        for i in [2,2,2,2]:
            self.f0.append(nn.Conv2d(ic, ic // i, 1, groups=ic // i))
            self.f1.append(nn.Conv2d(ic, ic // i, 1, groups=ic // i))
            self.f2.append(nn.Conv2d(ic, ic // i, 1, groups=ic // i))
            self.f3.append(nn.Conv2d(ic, ic // i, 1, groups=ic // i))
            self.f4.append(nn.Conv2d(ic, ic // i, 1, groups=ic // i))
            ic=ic//i

    def forward(self, x):

        out=[]

        x0=x
        for i in range(4):
            x0=channel_shuffle(x0,2)
            x0=self.f0[i](x0)
        out.append(x0)

        x1=x
        for i in range(4):
            x1=channel_shuffle(x1,4)
            x1=self.f1[i](x1)
        out.append(x1)

        x2=x
        for i in range(4):
            x2=channel_shuffle(x2,8)
            x2=self.f2[i](x2)
        out.append(x2)

        x3=x
        for i in range(4):
            x3=channel_shuffle(x3,16)
            x3=self.f3[i](x3)
        out.append(x3)

        x4=x
        for i in range(4):
            x4=channel_shuffle(x4,32)
            x4=self.f4[i](x4)
        out.append(x4)

        """
        for i in range(4):
            g=int(2*2**i)

            x0=channel_shuffle(x,g)
            out.append(self.f0[i](x0))

            x1=channel_shuffle(x,g)
            out.append(self.f1[i](x1))

            x2=channel_shuffle(x,g)
            out.append(self.f2[i](x2))

            x3=channel_shuffle(x,g)
            out.append(self.f3[i](x3))
        """

        for i in out:
            print(i.shape)

        b1=torch.cat(out,dim=1)

        b2=self.fc(x)
        print(b1.shape)
        print(b2.shape)
        out=torch.cat([b1,b2],dim=1)



        return out

class MIN512v2(nn.Module):
    def __init__(self):
        super(MIN512v2, self).__init__()
        self.f1=MIX512()
        self.f2=MIX512()


def MIXtest():
    from thop import profile
    i=512
    data=torch.randn(1,i,10,10)
    m=MIX512(i)

    flops = profile(m, inputs=(data,), verbose=False)[0] / 1E9 * 2  # stride GFLOPS
    print(flops)


if __name__ == '__main__':
    #netTest()

    #mutilspecsTest()

    testMainResBlockBackbone()

    pass

