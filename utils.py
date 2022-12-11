import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

# function to process input
def pre_pros_img(image,img_size):
    process = transforms.Compose([transforms.Resize(size=(img_size,img_size)),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x:x[torch.LongTensor([2,1,0])]),
                               transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                                                    std=[1,1,1]),
                               transforms.Lambda(lambda x:x.mul_(255))
                                  ])

    return process(image)

def post_pros_img(img):
    process = transforms.Compose([transforms.Lambda(lambda x:x.mul_(1/255.)),
                                  transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                                                       std=[1,1,1]),
                                  transforms.Lambda(lambda x:x[torch.LongTensor([2,1,0])]),
                                  ])

    return process(img)

post_pros_2 = transforms.Compose([transforms.ToPILImage()])
def postp(tensor):
    t = post_pros_img(tensor)
    t[t>1] = 1
    t[t<0] = 0
    img = post_pros_2(t)

    return img

def toGpu(func):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return func.to(device)
