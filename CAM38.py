import sys
sys.path.append("/media/old_ubuntu/media/mazhiming/L2G-main/")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch.nn.functional as F
from torchvision.io.image import read_image

from torchvision.models import resnet50
from models import resnet38
import models
from torchcam.methods import GradCAM
from torchcam.methods import CAM as get_cam_map
import importlib
import argparse
import torch
from tools.ai.torch_utils import *
from tools.ai.demo_utils import *
from tools.general.Q_util import *
import core.models as fcnmodel
import torch
import torch.nn as nn
import network.resnet38_base
from torchvision.transforms.functional import normalize, resize, to_pil_image


set_seed(0)
def get_arguments():
    parser = argparse.ArgumentParser(description='The Pytorch code of L2G')
    parser.add_argument("--num_classes", type=int, default=20)
    return parser.parse_args()


def vispcam(images,features,mask):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    cams = (make_cam(features)[:,:-1] * mask)
    obj_cams = cams.max(dim=1)[0]
    image = get_numpy_from_tensor(images)
    cam = get_numpy_from_tensor(obj_cams)[0]

    image = denormalize(image, imagenet_mean, imagenet_std)[..., ::-1]
    h, w, c = image.shape

    cam = (cam * 255).astype(np.uint8)
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    cam = colormap(cam)

    image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)[..., ::]
    image = image.astype(np.float32) / 255.
    return image

def cam_vis(model,input_tensor,name):
    savepth='CAMtest/cam/'+name
    cam,out1,convlist = model(input_tensor.unsqueeze(0))
    mask=torch.zeros_like(out1).squeeze()
    mask[out1.squeeze(0).argmax().item()]=1
    mask=mask.unsqueeze(0)
    img = vispcam(input_tensor,cam,mask.unsqueeze(2).unsqueeze(3))*255
    print("out1="+str(out1.squeeze(0).argmax().item()))
    cv2.imwrite(savepth,img)
    return out1
    
def sp_cam_vis(model,input_tensor,name,out1):
    savepth='CAMtest/sp_img_cam/'+name
    cam,out1,convlist = model(input_tensor)
    mask=torch.zeros_like(out1).squeeze()
    mask[out1.squeeze(0).argmax().item()]=1
    mask=mask.unsqueeze(0)
    img = vispcam(input_tensor.squeeze(0),cam,mask.unsqueeze(2).unsqueeze(3))*255
    print("out2="+str(out1.squeeze(0).argmax().item()))
    cv2.imwrite(savepth,img)
    return cam,mask

def camp_vis(map,Qs,input_tensor,mask):
    default_size_h,default_size_w=16,16
    b,c,h,w=map.size()
    # map = F.interpolate(map,size=(int(h/2),int(w/2)),mode='bilinear', align_corners=False)
    for i in range(50):
        sp_map = upfeat(map,Qs,default_size_h,default_size_w)
        map = poolfeat(sp_map, Qs,default_size_h,default_size_w)
    
    map = F.interpolate(map,size=(int(h*2),int(w*2)),mode='bilinear', align_corners=False)
    
    savepth='CAMtest/cam+/'+name

    img = vispcam(input_tensor.squeeze(0),map,mask.unsqueeze(2).unsqueeze(3))*255
    cv2.imwrite(savepth,img)
    return sp_map

args = vars(get_arguments())
#get model(pretrained)
model = getattr(importlib.import_module("network.resnet38"), 'Net')(args)
weights_dict = network.resnet38_base.convert_mxnet_to_torch('./models/ilsvrc-cls_rna-a1_cls1000_ep-0001.params')
weights_dict = torch.load('log/spimg_train_nni2022-09-15_19_54_41/best_checkpoint.pth')
model.load_state_dict(weights_dict, strict=False)

model.cuda().eval()

#get Qmodel
Q_model = fcnmodel.SpixelNet1l_bn().cuda()
Q_model.load_state_dict(torch.load('models_ckpt/Q_model_final.pth'))
Q_model = nn.DataParallel(Q_model)
Q_model.eval()

#get img
name='2008_000219.jpg'
img = read_image("CAMtest/img/"+name).cuda()
C,H,W=img.size()
input_tensor = normalize(resize(img, (448, 448)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


#get Q
Q = Q_model(input_tensor.unsqueeze(0))

#get sp_img
input_sp = poolfeat(input_tensor.unsqueeze(0), Q, 16, 16)
# input_sp = upfeat(input_sp, Q, 16, 16)
input_sp = F.interpolate(input_sp,(224,224),mode='bilinear', align_corners=False)

#vis
out1=cam_vis(model,input_tensor,name)
sp_cam,mask=sp_cam_vis(model,input_sp,name,out1)
camp_vis(sp_cam,Q,input_tensor,mask)






