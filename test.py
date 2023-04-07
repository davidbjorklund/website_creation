import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 23e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NOISE_DIM = 100
NUM_EPOCHS = 251
FEATURES_DISC = 64
FEATURES_GEN = 64


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # input: N x CHANNELS x 1 x 1
            self._block(channels_noise, features_g * 8, 4, 1, 0),
            # => 4x4
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            # => 8x8
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            # => 16x16
            self._block(features_g * 2, features_g, 4, 2, 1),
            # => 32x32
            nn.ConvTranspose2d(
                features_g, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # => 64x64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    # Initialize weights (according to the DCGAN paper)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
initialize_weights(gen)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

def load_checkpoint(checkpoint):
    gen.load_state_dict(checkpoint['state_dict'])
    opt_gen.load_state_dict(checkpoint['optimizer'])

t = input("Checkpoint #")

load_checkpoint(torch.load("./checkpoint/c-"+t+".pth.tar"))

while True:
    noise = torch.randn(1, 100, 1, 1).to(device)
    fake = gen(noise)

    img = fake[0].detach().cpu().permute(1, 2, 0).numpy()
    
    fig = plt.figure()
    plt.imshow(img)
    plt.title('Generated Data')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()
