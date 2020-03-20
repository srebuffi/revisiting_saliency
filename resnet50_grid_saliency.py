import torch
import torch.nn as nn
from torchvision import models
from torchvision import utils
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F


def change_relus(model, inplace=False):
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = inplace
    return model


class FeatureExtractor(object):
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
        self.fmap_pool = {}
        self.grad_pool = {}
        self.handlers = []

        def forward_hook(key):
            def forward_hook_(module, input, output):
                self.fmap_pool[key] = output.detach()
            return forward_hook_

        def backward_hook(key):
            def backward_hook_(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()
            return backward_hook_

        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.handlers.append(module.register_forward_hook(forward_hook(name)))
                self.handlers.append(module.register_backward_hook(backward_hook(name)))

    def __call__(self, x):
        return self.model(x)


class Extract_Aggregate(object):
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = FeatureExtractor(self.model, target_layer_names)
        if isinstance(target_layer_names, list):
            self.target_layer_names = target_layer_names
        else:
            self.target_layer_names = [target_layer_names]
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, phase1, phase2, index=None):
        if self.cuda:
            input = input.cuda()

        output = self.extractor(input)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros(output.shape, dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot)
        loss = -torch.sum(one_hot.cuda() * output)

        _ = torch.autograd.grad(loss, self.model.parameters())

        # Process all the targeted layers
        for num, target_layer_name in enumerate(self.target_layer_names):
            target = self.extractor.fmap_pool[target_layer_name].clone().cpu()
            grad_init = self.extractor.grad_pool[target_layer_name].clone().cpu()

            # Phase 1 of the framework (i.e. the possible types of virtual identity layers)
            if phase1 == 'conv1x1':
                out = -torch.matmul(target.permute(0, 2, 3, 1).view(-1, target.size(1), 1), grad_init.permute(0, 2, 3, 1).view(-1, 1, grad_init.size(1)))
                out = out.view(out.size(0), -1).permute(1, 0).view(1, -1, target.shape[2], target.shape[3])
            elif phase1 == 'conv3x3':
                unfold_act = F.unfold(target, kernel_size=3, padding=1)
                unfold_act = unfold_act.view(1, target.shape[1] * 9, target.shape[2], target.shape[3])
                out = -torch.matmul(unfold_act.permute(0, 2, 3, 1).view(-1, unfold_act.size(1), 1), grad_init.permute(0, 2, 3, 1).view(-1, 1, grad_init.size(1)))
                out = out.view(out.size(0), -1).permute(1, 0).view(1, -1, target.shape[2], target.shape[3])
            elif phase1 == 'conv3x3_depthwise':
                unfold_act = F.unfold(target, kernel_size=3, padding=1)
                unfold_act = unfold_act.view(1, target.shape[1], 9, target.shape[2], target.shape[3])
                out = -(unfold_act * grad_init[:, :, None]).view(1, -1, target.shape[2], target.shape[3])
            elif phase1 == 'scaling':
                out = -target * grad_init
            elif phase1 == 'bias':
                if target_layer_name == 'layer4':
                    continue  # the bias layer at the end of a ResNet gives a constant map
                out = -grad_init

            # Phase 2 of the framework (i.e. the possible types of aggregation)
            if phase2 == 'norm':
                out = torch.norm(out, 2, 1, keepdim=True)
            elif phase2 == 'filter_norm':
                out = torch.norm(torch.clamp(out, min=0), 2, 1, keepdim=True)
            elif phase2 == 'sum':
                out = torch.sum(out, 1, keepdim=True)
            elif phase2 == 'max':
                out = torch.max(out, 1, keepdim=True)[0]

            out = F.interpolate(out, size=(224, 224), mode='bilinear')
            out = out - torch.min(out)
            out = out / torch.max(out)

            # Accumulate different layers by multiplying the saliency maps with each other
            if num == 0:
                final_out = out.cpu().data.numpy()[0, 0]
            else:
                final_out *= out.cpu().data.numpy()[0, 0]

        return final_out


if __name__ == '__main__':
    # Load the model
    model = models.resnet50(pretrained=True)
    model = change_relus(model, False)
    model.eval()
    print(model)

    # Load the image
    img = Image.open('./examples/cat_dog.png')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    input = transform(img)[None]

    # Choose the target
    # target_index = 282  # the cat
    target_index = 242  # the dog

    # Choose the targeted layers
    layer_names = {
        "layer2": "layer2.0",  # After the first downsampling
        "layer3": "layer3.0",  # After the second downsampling
        "layer4": "layer4.0",  # After the third downsampling
        "end": "layer4",  # Before the Average Pooling layer
        "combi": ["layer2.0", "layer3.0", "layer4.0", "layer4"],  # Combining 4 layers in one pass
    }

    # Rows of the grid correspond to the possible phase 1 of the framework (i.e. the possible types of virtual identity layers)
    rows = ['scaling', 'conv1x1', 'conv3x3', 'bias']
    # Columns of the grid correspond to the possible phase 2 of the framework (i.e. the possible types of aggregation)
    cols = ['sum', 'filter_norm', 'norm', 'max']

    # Computing the grid of saliency maps
    for i, layer_name in enumerate(layer_names.keys()):
        f, ax = plt.subplots(len(rows), len(cols), figsize=(len(cols) * 4, len(rows) * 4))
        extrac_aggreg = Extract_Aggregate(model=model, target_layer_names=layer_names[layer_name], use_cuda=True)
        for row, phase1 in enumerate(rows):
            ax[row, 0].set_ylabel(phase1, fontsize=24)
            for col, phase2 in enumerate(cols):
                if layer_names[layer_name] == 'layer4' and phase1 == 'bias':
                    continue  # the bias layer at the end of a ResNet gives a constant map
                ax[0, col].set_xlabel(phase2, fontsize=24)
                ax[0, col].xaxis.set_label_position('top')
                mask = extrac_aggreg(input, phase1, phase2, target_index)
                ax[row, col].text(155, 25, layer_name, fontsize=24, bbox=dict(facecolor="white", alpha=0.5))
                ax[row, col].imshow(np.transpose(utils.make_grid(input.data.cpu(), normalize=True).numpy(), (1, 2, 0)))
                ax[row, col].imshow(np.float32(mask).squeeze(), cmap="jet", alpha=0.7)
                ax[row, col].set_xticks([])
                ax[row, col].set_yticks([])

        plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        plt.savefig(f"grid_resnet_{target_index}_{layer_name}.png", format="png", pad_inches=0.0)
