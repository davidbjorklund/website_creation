import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from skimage import io
from torch.utils.data import ( DataLoader, Dataset )
from torch.utils.tensorboard import SummaryWriter

class WebsiteDataSet(Dataset):
    def __init__(self, root_dir):
        # path to dataset csv file
        self.annotations = pd.read_csv(root_dir + "WebScreenshotsTourism.csv")
        self.root_dir = root_dir

    def __len__(self):
        return len(self.annotations) # 5000

    def __getitem__(self, index):
        # path to image from dataset
        img_path = os.path.join(self.root_dir, "screenshots-64x64/tourism/" + self.annotations.iloc[index, 0].replace("http://", "") + ".jpg")

        img = io.imread(img_path, pilmode='RGB')

        image = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(3)], [0.5 for _ in range(3)]
            ),
        ])(img)
        
        return (image, 1)


# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 9e-5
BATCH_SIZE = 32
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NOISE_DIM = 100
NUM_EPOCHS = 1000
FEATURES_DISC = 64
FEATURES_GEN = 64

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x CHANNELS x 64 x 64
            self._block(channels_img, features_d, 4, 2, 1),
            # => 32x32
            self._block(features_d, features_d * 2, 4, 2, 1),
            # => 16x16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            # => 8x8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # => 4x4
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            # => 1x1
            nn.Sigmoid(),
            # => 0/1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


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


def test():
    N, in_channels, H, W = NOISE_DIM, CHANNELS_IMG, IMAGE_SIZE, IMAGE_SIZE
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 64)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 64)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, tests passed!")


def save_checkpoint(state, t):
    filename = "./checkpoint/c-"+str(t)+".pth.tar"
    torch.save(state, filename)


dataset = WebsiteDataSet(root_dir = './data/')

train_set, test_set = torch.utils.data.random_split(dataset, [4800, 200])

dataloader = DataLoader(dataset = train_set, batch_size = BATCH_SIZE, shuffle = True)

gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")

if (not os.path.exists("checkpoint")):
    os.mkdir("checkpoint")

step = 0

if __name__ == "__main__":
    test()

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    j = 0
    for batch_idx, (real, _) in enumerate(dataloader):
        # prepare real and fake batch
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)
        
        # Train Discriminator
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses and print to tensorboard
        if j == 0:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}")
            if epoch >= NUM_EPOCHS/2 and epoch % int(NUM_EPOCHS / 20) == 0:
                checkpoint = {'state_dict': gen.state_dict(), 'optimizer': opt_gen.state_dict()}
                save_checkpoint(checkpoint, step)
                with torch.no_grad():
                    fake = gen(fixed_noise)
                    img_grid_fake = torchvision.utils.make_grid(fake[:16], normalize=True)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                step += 1
        j += 1