from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image, ImageFont, ImageDraw

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import os.path
import torchvision.utils
import time
device = torch.device('cuda:0')

# parser = argparse.ArgumentParser(description='PyTorch Neural-Style')
# parser.add_argument('--content', default='')
# parser.add_argument('--size', type=int, default=512)
# parser.add_argument('--style', default='')
# parser.add_argument('--output', default='')
# parser.add_argument('--class_type', default='Style')  # Artist
# parser.add_argument('--num_steps', type=int, default=300)
# parser.add_argument('--style_weight', type=float, default=1e6)
# parser.add_argument('--content_weight', type=float, default=1)
# parser.add_argument('--mode', default='file')  # folder
# parser.add_argument('--content_folder', default='')
# parser.add_argument('--style_folder', default='')
# parser.add_argument('--output_folder', default='')
# parser.add_argument('--sum_img_name', default='')
# parser.add_argument('--content_layers', type=str, nargs='+')
# parser.add_argument('--style_layers', type=str, nargs='+')
# parser.add_argument('--lr', type=float, default=1)

## for debug
parser = argparse.ArgumentParser(description='PyTorch Neural-Style')
parser.add_argument('--content', default='./image/0728/horse.jpg')
parser.add_argument('--size', type=int, default=512)
parser.add_argument('--style', default='./image/0728/tree.jpg')
parser.add_argument('--output', default='./image/0728/out_1e6.png')
parser.add_argument('--class_type', default='ori_vgg19')  # Artist
parser.add_argument('--num_steps', type=int, default=300)
parser.add_argument('--style_weight', type=float, default=1e6)
parser.add_argument('--content_weight', type=float, default=1)
parser.add_argument('--mode', default='file')  # folder
parser.add_argument('--content_folder', default='')
parser.add_argument('--style_folder', default='')
parser.add_argument('--output_folder', default='')
parser.add_argument('--sum_img_name', default='')
parser.add_argument('--content_layers', type=str, nargs='+', default=['1'])
parser.add_argument('--style_layers', type=str, nargs='+', default=['1', '2', '3', '4', '5'])
parser.add_argument('--lr', type=float, default=1)


args = parser.parse_args()
print(time.asctime(time.localtime(time.time())))
for name, value in vars(args).items():
    print('{} = {}'.format(name, value))

# desired size of the output image
imsize = args.size

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name).convert(mode='RGB')
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

print("=> creating model")
if args.class_type == 'Artist':
    model_path = '/media/gisdom/2TB_2/luyue/examples/imagenet/archive/artist/checkpoint_epoch87.pth.tar'
    vgg19 = models.vgg19()
    vgg19.classifier._modules['6'] = nn.Linear(4096, 23)
    print("=> loading state_dict".format(model_path))
    vgg19_data = torch.load(model_path)
    vgg19.load_state_dict(vgg19_data['state_dict'])
    cnn = vgg19.features.to(device).eval()
if args.class_type == 'Style':
    model_path = '/media/gisdom/2TB_2/luyue/examples/imagenet/archive/style/checkpoint_epoch37.pth.tar'
    vgg19 = models.vgg19()
    vgg19.classifier._modules['6'] = nn.Linear(4096, 18)
    print("=> loading state_dict".format(model_path))
    vgg19_data = torch.load(model_path)
    vgg19.load_state_dict(vgg19_data['state_dict'])
    cnn = vgg19.features.to(device).eval()
if args.class_type == 'ori_vgg19':
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# desired depth layers to compute style/content losses :
# content_layers_default = ['conv_4']
# style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

content_layers_default = ['conv_'+i for i in args.content_layers]
style_layers_default = ['conv_'+i for i in args.style_layers]


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            del target
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            del target_feature
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()], lr=args.lr)
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score

            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    model(input_img)
    style_score = 0
    content_score = 0
    for sl in style_losses:
        style_score += sl.loss
    for cl in content_losses:
        content_score += cl.loss
    style_score *= style_weight
    content_score *= content_weight

    return [input_img, content_score, style_score]



if args.mode == 'file':
    if not os.path.exists(os.path.dirname(args.output)):
        os.mkdir(os.path.dirname(args.output))
    style_img = image_loader(args.style)
    content_img = image_loader(args.content)
    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"
    input_img = content_img.clone()
    input_img = torch.randn(content_img.data.size(), device=device)
    output, content_loss, style_loss = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, num_steps=args.num_steps,
                                style_weight=args.style_weight, content_weight=args.content_weight)
    # torchvision.utils.save_image(output, args.output)

    to_pil = transforms.ToPILImage()
    output = to_pil(torch.squeeze(output.data).cpu())
    draw = ImageDraw.Draw(output)
    font = ImageFont.truetype("./simsun.ttc", 40)
    draw.text(xy=(0, 0), text='c_loss={:.1f}\ns_loss={:.1f}'.format(content_loss, style_loss),
              font=font, fill='#ffffff')

    img_sum = Image.new('RGB', (args.size*3, args.size), (255, 255, 255))
    img_sum.paste(Image.open(args.style).resize((args.size, args.size)), (0, 0))
    img_sum.paste(Image.open(args.content).resize((args.size, args.size)), (args.size, 0))
    img_sum.paste(output, (args.size*2, 0))
    img_sum.save(args.output)
    print('Save image {}'.format(args.output))

if args.mode == 'folder':
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    contents = os.listdir(args.content_folder)
    styles = os.listdir(args.style_folder)
    contents = [i for i in contents if not i.startswith('.')]
    styles = [i for i in styles if not i.startswith('.')]

    contents_path = args.content_folder
    styles_path = args.style_folder
    output_path = args.output_folder
    width = args.size + 120
    height = args.size + 120
    width_sum = width * (len(styles) + 1)
    height_sum = height * (len(contents) + 1)
    img_sum = Image.new('RGB', (width_sum, height_sum), (255, 255, 255))
    print('contents {}'.format(contents))
    print('styles {}'.format(styles))
    font = ImageFont.truetype("./simsun.ttc", 40)

    for i_content, content in enumerate(contents):
        for i_style, style in enumerate(styles):
            print('')
            print(time.asctime(time.localtime(time.time())))
            content_img = image_loader(os.path.join(args.content_folder, content))
            style_img = image_loader(os.path.join(args.style_folder, style))
            print('content: {}\t style: {}'.format(content, style))
            assert style_img.size() == content_img.size(), \
                "we need to import style and content images of the same size"
            input_img = content_img.clone()
            output, content_loss, style_loss = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                        content_img, style_img, input_img, num_steps=args.num_steps,
                                        style_weight=args.style_weight, content_weight=args.content_weight)
            output_name = os.path.splitext(content)[0] + '_' + os.path.splitext(style)[0] + '.png'
            torchvision.utils.save_image(output, os.path.join(args.output_folder, output_name))
            print('Save image {}'.format(output_name))

            to_pil = transforms.ToPILImage()
            output = to_pil(torch.squeeze(output.data).cpu())
            img_sum.paste(output, (i_style * width + width, i_content * height + height ))
            draw = ImageDraw.Draw(img_sum)
            draw.text(xy=(i_style * width + width , i_content * height + height - 80),
                      text='c_loss={:.1f}\ns_loss={:.1f}'.format(content_loss, style_loss),
                      font=font, fill='#000000')

    for i, content in enumerate(contents):
        img = Image.open(os.path.join(contents_path, content))
        img = img.resize((args.size, args.size))
        img_sum.paste(img, (0, i * height + height))

    for i, style in enumerate(styles):
        img = Image.open(os.path.join(styles_path, style))
        img = img.resize((args.size, args.size))
        img_sum.paste(img, (i * width + width, 0))

    img_sum.save(os.path.join(output_path, args.sum_img_name))
    print('Save image {}'.format(args.sum_img_name))

print(time.asctime(time.localtime(time.time())))
print('finish')
