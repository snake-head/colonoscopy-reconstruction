import os
import torch
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import style_transfer.utils as utils
from style_transfer.generators import define_Gen
import PIL.Image as Image


def style_transfer(img_path):
    """
    Transfer an image from real domain to virtual domain
    :param img_path: Path to image
    :return: Virtual domain image
    """

    transform = transforms.Compose(
        [transforms.Resize((480, 480)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    image0 = Image.open(img_path)
    if transform:
        if image0.mode != 'RGB':
            image0 = image0.convert('RGB')
        image0 = image0.crop((155, 0, 1162, 1007))  # Cut off the black edge
        image0 = transform(image0)

    Gba = define_Gen(input_nc=3, output_nc=3, ngf=64, netG='resnet_9blocks', norm='instance',
                     use_dropout=False, gpu_ids=[0])

    try:
        testmodelpath = './style_transfer/model/latest.ckpt'
        ckpt = utils.load_checkpoint('%s' % (testmodelpath))  # /latest.ckpt
        Gba.load_state_dict(ckpt['Gba'])
    except:
        print(' [*] No checkpoint!')

    Gba.eval()

    with torch.no_grad():
        a_real_test = utils.cuda(image0).unsqueeze(0)
        b_fake_test = Gba(a_real_test)
        pic = (b_fake_test.data + 1) / 2.0
        #pic =  b_fake_test.data
        # torchvision.utils.save_image(pic, './style_transfer/' + '/sample.jpg', nrow=1)
    return pic.squeeze()
