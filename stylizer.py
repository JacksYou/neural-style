import torch
import torchvision
import copy
import skimage.io
import skimage.transform

from tqdm import tqdm
import PIL as pillow
import numpy as np
import matplotlib.pyplot as plt

class NeuralStylizer:


    def __init__(self, cnn, content_img, style_imgs,  style_weights, target_shape, output_shape,
                 backend, content_layers, style_layers, pooling, A, B, L):
        self.dtype = torch.cuda.FloatTensor if backend == 'cuda' else torch.FloatTensor

        self.imsize = target_shape[0]
        self.content_img, self.original_sz = self._load(content_img, target_shape)
        self.content_img = self.content_img.type(self.dtype)

        cnn = copy.deepcopy(cnn)

        conv_layers = [(1,1), (1,2),
                       (2,1), (2,2),
                       (3,1), (3,2), (3,3), (3,4),
                       (4,1), (4,2), (4,3), (4,4),
                       (5,1), (5,2), (5,3), (5,4),]

        content_losses = []
        style_losses = []
        tv_losses = []
        model = torch.nn.Sequential()  # the new Sequential module network

        if backend == 'cuda':
            model = model.cuda()

        i = 0
        j = 1
        # apppend total variation at the beginning of model
        tv_loss = TVLoss(L)
        model.add_module("tv_loss", tv_loss)
        tv_losses.append(tv_loss)

        for layer in list(cnn):
            if isinstance(layer, torch.nn.Conv2d):
                name = "conv_{}_{}".format(conv_layers[i][0], conv_layers[i][1])
                model.add_module(name, layer)

                if name in content_layers:
                    target = model(self.content_img).clone()
                    content_loss = ContentLoss(target, A)
                    model.add_module("content_loss_" + str(i), content_loss)
                    content_losses.append(content_loss)

                if name in style_layers:
                    for idx, style_img in enumerate(style_imgs):
                        img, _ = self._load(style_img, target_shape)
                        target = model(img.type(self.dtype)).clone()
                        style_loss = StyleLoss(target, style_weights[idx]*B)
                        model.add_module("style_loss_{}_{}".format(i, idx), style_loss)
                        style_losses.append(style_loss)

            if isinstance(layer, torch.nn.ReLU):
                name = "relu_{}_{}".format(conv_layers[i][0], conv_layers[i][1])
                model.add_module(name, layer)
                i += 1

            if isinstance(layer, torch.nn.MaxPool2d):
                name = "pool_" + str(j)
                if pooling == 'avg':
                    layer = torch.nn.AvgPool2d(layer.kernel_size,  layer.stride, layer.padding, layer.ceil_mode)
                model.add_module(name, layer)
                j += 1

        self.model = model
        self.style_losses = style_losses
        self.content_losses = content_losses
        self.tv_losses = tv_losses

    def __call__(self, iterations, output_shape, init, output_file):
        sz, c, h, w = self.content_img.size()
        output_shape = self.original_sz if output_shape is None else output_shape

        input_img = torch.autograd.Variable(torch.randn((sz, c, h, w))).type(self.dtype) if init == 'random' else self.content_img.clone()

        input_param = torch.nn.Parameter(input_img.data)
        optimizer = torch.optim.LBFGS([input_param])

        with tqdm(total=iterations) as pbar:
            run = [0]
            while run[0] < iterations:

                def closure():
                    input_param.data.clamp_(0, 1)

                    optimizer.zero_grad()
                    self.model(input_param)
                    style_score = 0
                    content_score = 0
                    tv_score = 0

                    for sl in self.style_losses:
                        style_score += sl.backward()
                    for cl in self.content_losses:
                        content_score += cl.backward()
                    for tl in self.tv_losses:
                        tv_score += tl.backward()

                    run[0] += 1
                    pbar.set_description('style err: {:4f} content err: {:4f} tv err: {:4f}'.format(
                          style_score.data[0], content_score.data[0],  tv_score.data[0]))
                    pbar.update(1)
                    return style_score + content_score + tv_score

                optimizer.step(closure)

            input_param.data.clamp_(0, 1)

        self._save(input_param.data, output_shape, output_file)

    def _load(self, img, dim):
        importer = torchvision.transforms.Compose([
                 torchvision.transforms.Resize(size=dim),
                 torchvision.transforms.ToTensor(),
                 ])
        image = pillow.Image.open(img)
        w, h = image.size
        data = np.array(image)
        data = data[:, :, [2, 1, 0]]
        image = pillow.Image.fromarray(data)
        image = torch.autograd.Variable(importer(image))
        image = image.unsqueeze(0)
        return image, (h, w)

    def _save(self, tensor, dim, name='result.jpg'):
        exporter = torchvision.transforms.Compose([
                   torchvision.transforms.ToPILImage(),
                   torchvision.transforms.Resize(size=dim),
                   ])
        image = tensor.clone().cpu()
        image = image.view(3, self.imsize, self.imsize)  # remove the fake batch dimension
        image = exporter(image)
        image = np.array(image)
        image = image[:, :, [2, 1, 0]]
        image = pillow.Image.fromarray(image)
        image.save(name)


class ContentLoss(torch.nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = torch.nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

class GramMatrix(torch.nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

class StyleLoss(torch.nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach()
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = torch.nn.MSELoss()

    def forward(self, tensor):
        self.output = tensor.clone()
        G = self.gram(tensor).mul_(self.weight)
        A = self.gram(self.target).mul_(self.weight)
        self.loss = self.criterion(G, A)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

class TVLoss(torch.nn.Module):

    def __init__(self, weight):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, tensor):
        self.output = tensor.clone()
        sz, c, h, w = tensor.size()

        h_tv = torch.sum(torch.pow(tensor[:,:,1:,:] - tensor[:,:,:h-1,:], 2))
        w_tv = torch.sum(torch.pow(tensor[:,:,:,1:] - tensor[:,:,:,:w-1], 2))
        self.loss = self.weight * torch.sqrt((w_tv + h_tv))
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
