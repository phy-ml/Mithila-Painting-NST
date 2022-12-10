# command to download weights
import os
import urllib.request

def download():
    if not os.path.exists('model_weight'):
        os.makedirs('model_weight')

    url = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"

    urllib.request.urlretrieve(url,"model_weight/vgg19-dcbb9e9d.pth")
    print('Download finished')

