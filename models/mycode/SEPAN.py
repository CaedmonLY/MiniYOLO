import torch
import torch.nn as nn

from models.mycode.effectlite import BasicResBlock,BasicResBlockIO
from models.mycode.effectlite import EfficientNet_lite


from models.common import BottleneckCSP

class ResBlockION(nn.Module):
    def __init__(self,channel,innerchannel,outchannel,n):
        super(ResBlockION, self).__init__()
        self.repeat=n-1
        self.nblock=[]
        self.res=BottleneckCSP(channel,outchannel)
        for i in range(n-1):
            self.nblock.append(BottleneckCSP(outchannel,outchannel))

    def forward(self,x):

        x=self.res(x)
        for i in range(self.repeat):
            x=self.nblock[i](x)
        return x



class ResBlockN(nn.Module):
    def __init__(self,channel,outchannel,n):
        super(ResBlockN, self).__init__()
        self.repeat=n
        self.nblock=BottleneckCSP(channel,outchannel,n)

    def forward(self,x):
        x=self.nblock(x)
        return x









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

class SELayers(nn.Module):
    def __init__(self,channel, reduction=16):
        super(SELayers, self).__init__()
        #self.SEs=[]
        self.SEs=torch.nn.Sequential()

        for i in range(6):
            self.SEs.add_module("ses"+str(i),SELayer(channel,reduction))
            #self.SEs.append(SELayer(channel,reduction))


    def forward(self,x1,x2):
        out=[]
        for i in range(3):
            x1[i]=self.SEs[i]  (x1[i])
            x2[i]=self.SEs[i+3](x2[i])
            out.append(x1[i]+x2[i])
        return out




from models.mycode.pan import PAN


class SEPANs(nn.Module):
    def __init__(self,channel=128):
        super(SEPANs, self).__init__()
        self.c0 = ResBlockN(channel, 96, 4)
        self.c1 = ResBlockN(channel, 96, 4)
        self.c2 = ResBlockN(channel, 96, 4)

        self.pan=PAN([96,96,96],96,3)
        self.selayers=SELayers(channel)

    """
    感觉应该是x先经过几个res后再进行pan，两个结果cat+
    """

    def forward(self,x):
        print('************************')
        return x

        x=self.pan(x)


        y=[]


        y.append(self.c0(x[0]))
        y.append(self.c1(x[1]))
        y.append(self.c2(x[2]))


        out=self.selayers(x,y)
        return out


from models.mycode.backbone import C3s


"""
当前2.3m
一个c3。   1117k，新增参数都在这里

"""
class SEPANsC3(nn.Module):
    def __init__(self,c1=116, c2=96, n=3, shortcut=True, g=1, e=0.5):
        super(SEPANsC3, self).__init__()
        self.C3b=C3s(c1, c2, n, shortcut, g, e)
        self.pan=PAN([c2, c2, c2], c2, 3)
        self.C3a=C3s(c2, c2, n, shortcut, g, e)
        self.selayers = SELayers(c2)

    def forward(self, x):
        x=self.C3b(x)
        y=self.pan(x)
        y=self.C3a(y)
        out=self.selayers(x,y)
        return out


from models.common import C3






class SEPANsC3T2(nn.Module):
    def __init__(self,c0=116,c1=116,c2=116, co=96, n=3, shortcut=True, g=1, e=0.5):
        super(SEPANsC3T2, self).__init__()

        """
        self.c0 = ResBlockN(c0,co,1 )
        self.c1 = ResBlockN(c1,co,1 )
        self.c2 = ResBlockN(c2,co,1 )

        """
        self.c0=C3(c0, co, 1, shortcut, g, e)
        self.c1=C3(c1, co, 1, shortcut, g, e)
        self.c2=C3(c2, co, 1, shortcut, g, e)


        self.C3b = C3s(co, co, n, shortcut, g, e)
        self.pan=PAN([co, co, co], co, 3)

        self.C3a = C3s(co, co, n, shortcut, g, e)
        self.selayers = SELayers(co)
        self.C3out=C3s(co, co, 3, shortcut, g, e)


    def forward(self, x):

        y=[]


        y.append(self.c0(x[0]))
        y.append(self.c1(x[1]))
        y.append(self.c2(x[2]))


        x=self.C3a(y)
        y=self.C3b(y)
        y=self.pan(y)

        out=self.selayers(x,y)
        out = self.C3out(out)

        return out



def testc():
    d=[
        torch.randn(1,128,32,32),
        torch.randn(1, 256, 32, 32),
        torch.randn(1, 512, 32, 32),
    ]
    m=SEPANsC3T2(128,256,512,96)
    for i in range(3):
        d[i]=d[i].cuda()


    m=m.cuda()
    #torch.save(m.state_dict(),'SEPANsC3t2')

    o=m(d)
    for i in o:
        print(i.shape)



class sepanbackbone(nn.Module):
    def __init__(self):
        super(sepanbackbone, self).__init__()
        self.backbone = EfficientNet_lite()
        self.sepan=SEPANs(96)

    def forward(self,x):
        x=self.backbone(x)
        print("sepanbackbone")
        for i in x:
            print(i.shape)
        #x=self.sepan(x)
        return x



if __name__ == '__main__':

    testc()

    #torch.save(C3s(96, 96, 3, True, 1, 0.5),"sdsd")
    pass