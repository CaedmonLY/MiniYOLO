
import torch.nn as nn
import torch

#from .ShuffleNetV2CP import shufflenet_v2_x0_5 as shufflenet
from models.mycode.ShuffleNetV2CP import shufflenet_v2_x1_0 as shufflenet

from models.mycode.pan import PAN



from models.common import C3

class C3s(nn.Module):

    def __init__(self,c1=116, c2=116, n=3, shortcut=True, g=1, e=0.5):
        super(C3s, self).__init__()

        self.c31=C3(c1, c2, n, shortcut, g, e)
        self.c32=C3(c1, c2, n, shortcut, g, e)
        self.c33=C3(c1, c2, n, shortcut, g, e)

    def forward(self, x):

        x1=self.c31(x[0])
        x2=self.c32(x[1])
        x3=self.c33(x[2])

        return [x1,x2,x3]


from models.mycode.myBackbone import DWPAN

class backbone(nn.Module):

    def __init__(self):
        super(backbone, self).__init__()
        self.base=shufflenet()
        #self.pan=PAN([48, 96, 192],96,3)           #与shufflenet_v2_x0_5 输出相匹配  48, 96, 192
        self.pan = PAN([116, 232, 464], 116, 3)

        self.c3=C3s()

        self.pan2=PAN([116,116,116],116,3)

        self.c3b=C3s()

        # self.pan = DWPAN([116, 232, 464], 116, 3)
        # self.pan2 = DWPAN([116, 116, 116], 116, 3)

        self.dwpan=DWPAN([116, 232, 464], 116, 3)


    def forward(self, x):
        x=self.base(x)

        x=self.pan(x)

        #x=self.c3(x)
        #x=self.pan2(x)
        #x=self.c3b(x)

        return x


def test():
    data=torch.randn(1,3,416,416)
    m=backbone()
    out=m(data)

    for i in out:
        print(i.shape)


    #torch.save(m,"tmp.pt")



if __name__ == '__main__':

    #test()
    data=torch.randn(1,3,416,416)
    b=backbone()


    pass