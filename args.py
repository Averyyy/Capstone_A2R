import os


class Args:
    cuda = True
    shuffle = True
    batch_size = 2
    image_shape = (3, 256, 256)
    learning_rate = 1e-3
    learning_rate_decay = 0.99
    weight_decay = 5e-4

    log_dir = 'vae_mse_loss'

    checkpoint_dir = './checkpoint/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    start_epoch = 0
    num_epochs = 10
    test_every = 2



if __name__ == "__main__":
    a = Args()
    print(a.cuda)
