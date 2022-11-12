from torch import nn
import torch

# from torchvision import models
# vgg16 = models.vgg16(pretrained=True)


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss = nn.MSELoss() #nn.CrossEntropyLoss()

    def forward(self, recon_x, x, mu, logvar):
        # x = x * 255
        # x.data = x.data.int().long().view(-1)
        # print(recon_x.shape)
        # recon_x = recon_x.permute(0, 2, 3, 4, 1)  # N * C * W * H
        # print(recon_x.shape)
        # recon_x = recon_x.contiguous().view(-1, 256)

        MSE = self.loss(recon_x, x)

        # see Appendix B from VAE paper:    https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD
