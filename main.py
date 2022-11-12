from torch.utils.data import DataLoader
import torch
from dataset import MyImageSet
from tqdm import tqdm
from vae_model import VAE
from train import Trainer
from loss import Loss
from args import Args


imgset_train = MyImageSet('./data/train')
imgset_test = MyImageSet('./data/test')
# parser = argparse.ArgumentParser(description='VAE MSE loss')

# args = parser.parse_args()
args = Args()
ImgLoader_train = DataLoader(imgset_train, batch_size=args.batch_size)
ImgLoader_test = DataLoader(imgset_test, batch_size=args.batch_size)

mse_loss = Loss()
model = VAE()


args.cuda = args.cuda and torch.cuda.is_available()

trainer = Trainer(model, mse_loss, ImgLoader_train, ImgLoader_test, args)
trainer.train()
