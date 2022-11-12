import numpy as np
import torch
from torch import nn


class _downsample(nn.Module):
    def __init__(self, c):
         # Encoder
        super(_downsample, self).__init__()
        self.conv1 = nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3)))#.view(-1, 8 * 8 * 16)
        return conv4


class _upsample(nn.Module):
    def __init__(self, c):
        # Decoder
        super(_upsample, self).__init__()
        self.conv5 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        conv5 = self.relu(self.bn5(self.conv5(x)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        return self.conv8(conv7)#.view(-1, 3, 32, 32)
    


class VAE(nn.Module):

    def __init__(self, shape=(3, 256, 256)):
        super(VAE, self).__init__()
        # self.shape = shape
        # self.img_input = inputs
        # self.alpha = alpha
        # self.beta = beta
        # self.gstep = 0
        # self.vgg_layers = vgg_layers
        # self.learning_rate = learning_rate
        # Encoder
        self.encoder_conv_net = _downsample(shape[0])
        self.out_size_encoder_conv_net = self._get_conv_out(self.encoder_conv_net, shape)
        self.fc1 = nn.Linear(int(np.prod(self.out_size_encoder_conv_net)), 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc21 = nn.Linear(512, 512)
        self.fc22 = nn.Linear(512, 512)
        self.relu = nn.ReLU()

        # Decoder
        self.fc3 = nn.Linear(512, 512)
        self.fc_bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, int(np.prod(self.out_size_encoder_conv_net)))
        self.fc_bn4 = nn.BatchNorm1d(int(np.prod(self.out_size_encoder_conv_net)))
        self.decoder_cov_net = _upsample(shape[0])

    def encoder(self, x):
        conv4 = self.encoder_conv_net(x)
        fc1 = self.relu(self.fc_bn1(self.fc1(conv4.view(-1, int(np.prod(self.out_size_encoder_conv_net))))))
        return self.fc21(fc1), self.fc22(fc1)

    def decoder(self, z):
        _, c, rows, cols = self.out_size_encoder_conv_net
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, c, rows, cols)
        return self.decoder_cov_net(fc4)

    def _get_conv_out(self, model, shape):
        o = model(torch.zeros(1, *shape))
        return o.size()
        # return int(np.prod(o.size()))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    # def _get_weights(self, name, shape):
    #     pass

    # def _get_biases(self, name, shape):
    #     pass

    # def _conv2d_bn_relu(self, inputs, name, kernel_size, in_channel, out_channel, stride, activation=True, bn=True):
    #     pass


    # def calculate_loss(self):
    #     pass

    # def optimize(self):
    #     pass

    # def build_model(self):
    #     pass 

if __name__ == "__main__":
    test = _downsample(3)
    img = torch.zeros(1, *[3, 256, 256])
    out = test(img)
    print(int(np.prod(out.size())))

    z = torch.rand(2, 512)

    # image_size = 256
    test2 = _upsample(3)

    fc3 = torch.nn.Linear(512, 512)
    fc_bn3 = torch.nn.BatchNorm1d(512)
    fc4 = torch.nn.Linear(512, 3*256*256)
    fc_bn4 = torch.nn.BatchNorm1d(3*256*256)
    Relu = torch.nn.ReLU()
    x = Relu(fc_bn3(fc3(z)))
    print(x.size())
    # x = Relu(fc_bn3(fc3(z)))
    y = Relu(fc_bn4(fc4(x))).view(-1, 3, 256, 256)
    print(y.size())
    out = test2(y)
    print(out.size())