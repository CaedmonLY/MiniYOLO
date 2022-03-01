

import torch
import cv2
import numpy as np

class randBoost(torch.nn.Module):
    """
    数据增强，继承nn.Module，以便可以在gpu上运行                                         cpu上也能执行

    目前不打算将它纳入到DataSet类里面处理。     直接对一个gpu中的一个batch处理
    这样的话，和以往框架有点不同的时，数据加载完后，还需进行    “再处理”                       直接在cpu上怕有点慢


    输入一个“预处理后的图像”和一个对象mask。
    对象以外的背景用0均值随机数替代。

    """
    def __init__(self,lablepath="J:/voc/VOCdevkit/voc/labels/"):
        self.lablepath=lablepath

    def __call__(self,standardization,batchimg,imgpaths,istrain=False):

        print(batchimg.shape)

        """
        standardization:  数据是否标准化到[-1,1]
        大多数论文采用[0,1]范围

        imgpath: 'J:\\voc\\VOCdevkit\\voc\\images\\val\\2008_000119.jpg'
        maskpath: J:\\voc\VOCdevkit\\voc\\labels\\trainMask
        """
        imgset="val"
        if(istrain):
            imgset="train"
        file=[]
        for path in imgpaths:
            mask=self.lablepath+imgset+"Mask/"+path[-15:-4]+".png"
            print(mask)
            mask=cv2.imread(mask,cv2.IMREAD_GRAYSCALE)
            mask=torch.from_numpy(mask)
            file.append(mask)
        masks=torch.stack(file,dim=0)
        return self.standBoost(standardization,batchimg,masks)

        pass

    def standBoost(self,standardization,batchimg_tensor,batchmask_tensor):
        """
        batchimg :   [batchsize,3,h,w]         float
        btachmask:   [batch,h,w]               float

        return  :    [batch,3,h,w]
        """
        batchimg=batchimg_tensor.float()
        batchmask=batchmask_tensor.float()
        batchmask = torch.stack((batchmask, batchmask, batchmask), 1)      # 也构造成三通道，索引的时候直接和img匹配

        shape=batchimg.shape
        if(torch.cuda.is_available()):
            boostTensor=torch.randn(shape).cuda()               # 标准正态分布中，大于3.9的部分，概率基本为0了。可以近似认为抽出来的数大致在[-3.9,3.9]
        else:
            boostTensor = torch.randn(shape)
        #boostTensor=boostTensor/3.9
        boostTensor = boostTensor / 3.9

        if(not standardization):
            boostTensor=(boostTensor+1)/2


        """        
        print("debug boost")
        print(batchimg.shape)
        print(batchmask.shape)
        print(boostTensor.shape)
        """

        #为了本地测试，传进来的数是   [batch,h,w,c],[batch,h,w]      *255是为了cv2演示效果
        #boostTensor=boostTensor*255
        #boostTensor=torch.abs(boostTensor)

        background=torch.where(batchmask<0.5)         #  float数，还是确定一个范围比较保险，选出 ==0 的index


        batchimg[background]=boostTensor[background]

        return batchimg


    def tensor2img(self,data,standardization=True):
        testpic = data.cpu()
        # chw->hwc
        pic = testpic.permute(0, 2, 3, 1).contiguous().float().numpy()
        if(standardization):
            pic = (pic + 1) * 255 / 2
        else:
            pic=pic*255
        pic = np.uint8(pic[0])  # 注意  pyotrch图片  是带batch的          使用opencv注意去掉batch
        return pic



class getOneBoostedImg(torch.nn.Module):
    """
    不太好在yolov5中嵌入，直接提前生成好了

    """
    def __init__(self,lablepath="J:/voc/VOCdevkit/voc/labels/newMask/"):
        self.lablepath=lablepath



    def __call__(self,istrain,img,imgpath):


        img=torch.from_numpy(img).float()
        mask = self.lablepath + imgpath[-15:-4] + ".png"
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        mask = torch.from_numpy(mask).float()


        shape=img.shape

        boostTensor=torch.randn(shape)               # 标准正态分布中，大于3.9的部分，概率基本为0了。可以近似认为抽出来的数大致在[-3.9,3.9]
        boostTensor = boostTensor / 3.9
        boostTensor=(boostTensor+1)/2*255
        #boostTensor = boostTensor* 255
        background=torch.where(mask<30)         #  float数，还是确定一个范围比较保险，选出 ==0 的index

        img[background]=boostTensor[background]


        img=np.uint8(img.numpy())

        return img





if __name__ == '__main__':


    file=[
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2010_003603.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2009_003189.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2009_001945.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2011_000785.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2010_002175.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2010_001837.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2008_006991.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2009_001909.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2011_000487.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2008_003331.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2011_002159.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2010_002645.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2009_003821.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2008_003681.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2008_002021.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2011_003079.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2009_003655.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2009_001607.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2008_006743.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2011_000213.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2008_003239.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2009_002185.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2010_002223.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2008_004575.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2008_001589.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2008_002971.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2009_003343.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2010_005849.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2010_000993.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2008_005735.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2010_001635.jpg',
        'J:\\voc\\VOCdevkit\\voc\\images\\val\\2010_003779.jpg'
    ]
    trans=prerandBoost()
    for i in file:
        trans(False,i)


    """
    for i in file:
        path=[i]

        boost = randBoost()
        img = cv2.imread(path[0])
        print(img.shape)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        c, h, w = img.shape
        img = img.resize(1, c, h, w)
        img = img.cuda()
        img = img / 255

        out = boost(False, img, path)

        img = boost.tensor2img(img, False)
        cv2.imshow("s", img)
        cv2.waitKey(0)
    """




