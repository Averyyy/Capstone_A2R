from torch import nn
import torch
from torchvision import models
import torchvision.transforms as T
# vgg16 = models.vgg16(pretrained=True)
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

# vgg16 = models.vgg16()
# print(vgg16)
## get the names of layers
# train_nodes, eval_nodes = get_graph_node_names(models.vgg16())
# print(eval_nodes)
return_nodes = {
    'features.1': 'feature_map64',
    'features.6': 'feature_map128',
    'features.11': 'feature_map256',
    #'features.18': 'feature_map512_1',
    #'features.25': 'feature_map512_2'
}

# features = create_feature_extractor(vgg16, return_nodes)
# inp = torch.randn(2, 3, 224, 224)
# with torch.no_grad():
#     f = features(inp)
# print(f.keys())

class Vgg16featuremap(torch.nn.Module):

    def __init__(self, return_nodes):
        super(Vgg16featuremap, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.body = create_feature_extractor(vgg16, return_nodes)

    def forward(self, x):
        with torch.no_grad():
            features = self.body(x)
        return features


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        # self.mseloss = nn.MSELoss() #nn.CrossEntropyLoss()
        self.transform = T.Resize(224)
        self.vgg_real = Vgg16featuremap(return_nodes)
        self.vgg_gen = Vgg16featuremap(return_nodes)

    def forward(self, recon_x, x, mu, logvar):
        # x = x * 255
        # x.data = x.data.int().long().view(-1)
        # print(recon_x.shape)
        # recon_x = recon_x.permute(0, 2, 3, 4, 1)  # N * C * W * H
        # print(recon_x.shape)
        # recon_x = recon_x.contiguous().view(-1, 256)
        recon_x = self.transform(recon_x)
        x = self.transform(x)
        # MSE = self.mseloss(recon_x, x)
        recon_f = [v for k, v in self.vgg_gen(recon_x).items()]
        real_f = [v for k, v in self.vgg_real(x).items()]
        # print(real_f[0])
        preceptual_loss = torch.sum(torch.square(recon_f[0] - real_f[0]))
        for i in range(1, len(real_f)):
            preceptual_loss += torch.sum(torch.square(recon_f[i] - real_f[i]))
        preceptual_loss = torch.mean(preceptual_loss)

        # see Appendix B from VAE paper:    https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return preceptual_loss + KLD


if __name__ == "__main__":
    loss = Loss()
    recon_x = torch.rand(2, 3, 224, 224)
    x = torch.rand(2, 3, 224, 224)
    mu = torch.rand(1, 10)
    logvar = torch.rand(1, 10)
    print(loss(recon_x, x, mu, logvar))
