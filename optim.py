from PIL import Image
from loss.loss import *
from model.vgg_19 import *
from utils import *
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt


class TransferStyle:
    def __init__(self, content, style, img_size=512):
        # load image into tensor, add extra dim for computation and shift to gpu
        self.content = toGpu(pre_pros_img(image=content,img_size=img_size).unsqueeze(0))
        self.style = toGpu(pre_pros_img(image=style,img_size=img_size).unsqueeze(0))

        # load model
        self.load_model()

        # copy the content from content image with grads for optimization img
        self.opt_img = Variable(self.content.data.clone(), requires_grad=True)

    def __call__(self,content_layer, style_layer, iter):
        self.train(content_layer=content_layer,
                   style_layer=style_layer,
                   iter=iter)

    def load_model(self):
        self.model = load_vgg19()

    def contenttarget(self,layer_name):
        target = [i.detach() for i in self.model(self.content, layer_name)]
        return target

    def styletarget(self, layer_name):
        target = [GramMatrix()(i).detach() for i in self.model(self.style, layer_name)]
        return target

    def target(self, content_layer, style_layer):
        style = self.styletarget(layer_name=style_layer)
        content = self.contenttarget(layer_name=content_layer)
        return style + content

    def loss_fun(self, content_layer, style_layer):
        loss_layer = style_layer + content_layer
        loss_fun_layer = [GramLoss()]*len(style_layer) + [nn.MSELoss()]*len(content_layer)
        return {'loss_layer':loss_layer,
                'loss_func':loss_fun_layer}

    def get_loss_weights(self):
        style_weight = [1e3/n**3 for n in [64,128,256,512,512]]
        content_weight = [1e0]
        print(sum(style_weight))
        return style_weight + content_weight

    def train(self, iter, style_layer, content_layer):
        target = self.target(content_layer=content_layer,style_layer=style_layer)

        #get the loss func
        loss = self.loss_fun(content_layer=content_layer,style_layer=style_layer)
        loss_layers = loss['loss_layer']
        loss_func = loss['loss_func']

        # get the weights for respective layers
        weights = self.get_loss_weights()

        # get the target for each layer
        target = self.target(content_layer=content_layer,style_layer=style_layer)

        # define the optimizer
        optimizer = optim.LBFGS([self.opt_img])
        max_iter = iter
        print_iter = 50
        n_iter = [0]

        while n_iter[0] <= max_iter:

            def closure():
                # set the grads to be zero
                optimizer.zero_grad()

                # get the output from the model
                out = self.model(self.opt_img, loss_layers)

                # get loss from respective sources
                loss = [weights[i]*loss_func[i](x, target[i]) for i,x in enumerate(out)]

                # sum the loss
                total_loss = sum(loss)
                total_loss.backward()

                n_iter[0] += 1

                # print the loss
                if n_iter[0] % print_iter == (print_iter - 1):
                    print(f"Iter :{n_iter[0]} ||  Loss :{total_loss.item()}")

                return total_loss

            optimizer.step(closure)

        # get the image out
        output_img = postp(self.opt_img.data[0].cpu().squeeze())
        # output_img = output_img.permute(1,2,0).numpy()
        # show image
        plt.imshow(output_img)
        plt.show()

        # save image
        output_img.save(r'Images/output/content_2_style_3.png')


if __name__ == "__main__":
    # load image
    style_img = Image.open(r"Images/style/style_3.png")
    content_img = Image.open(r"Images/content/content_2.png")

    style_layer = ['r11','r21','r31','r41', 'r51']
    content_layer = ['r42']
    iter = 200

    target = TransferStyle(content=content_img,style=style_img)
    final = target(iter=iter, content_layer=content_layer, style_layer=style_layer)

