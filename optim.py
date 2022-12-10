import torch
from PIL import Image
from loss.loss import *
from model.vgg_19 import *
from utils import *
from torch.autograd import Variable


# step 1: load the vgg
# step 2: switch off the gradients for vgg
# step 3: load model into gpu
# step 4: load the content and style image
# step 5: copy content image with grads ON
# step 6: list the style and content image layer names
# step 7: compute the vgg output for the style and content layers and store as target
# step 8: list the weights attached with each layer
# step 9: set up the optimizer
# step 10: compute the loss from the optim image
# step 11: perform backprop on the loss output

class TransferStyle:
    def __init__(self, content, style, img_size=512):
        # load image into tensor, add extra dim for computation and shift to gpu
        self.content = toGpu(pre_pros_img(image=content,img_size=img_size).unsqueeze(0))
        self.style = toGpu(pre_pros_img(image=style,img_size=img_size).unsqueeze(0))

        # load model
        self.load_model()

        # copy the content from content image with grads for optimization img
        self.opt_img = Variable(self.content.data.clone(), requires_grad=True)

    def __call__(self, *args, **kwargs):
        pass

    def load_model(self):
        self.model = load_vgg19()

    def contentloss(self):
        pass

    def styleloss(self):
        pass

if __name__ == "__main__":
    # load image
    style_img = Image.open(r'Images/style.png')
    content_img = Image.open(r'Images/content.png')

    target = TransferStyle(content=content_img,style=style_img)
    print(target.style.requires_grad)
    print(target.content.requires_grad)
    print(target.opt_img.requires_grad)
    print([i.requires_grad for i in target.model.parameters()])

