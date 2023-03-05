# Website Creation
Code for high-school graduation project:
> ## "Website creation using generative adversarial networks"
> **A study in the method of generating websites using AI and its effect on business**
> By: [David Björklund](https://github.com/davidbjorklund)

> At: [Värmdö Gymnasium](https://www.vgy.se/)

## Summary
A DCGAN to create images of websites, using modified dataset from:
Fahri Aydos. (2020). *WebScreenshots* [Data set]. [Kaggle](https://doi.org/10.34740/KAGGLE/DS/202248).

## Example results
Fake:
![Fake Gif](/result/fake.gif)
![Fake PNG](/result/fake_29.png)
Real:
![Real PNG](/result/real.png)

## Requirements
- Python (Recommended 3.7+)
    - Pytorch
    - Torchvision
    - Sci-kit
- Processing power (CPU)
    - Recommended GPU

## Usage
Start training:
> python dcgan.py

Start tensorboard (after training, to see results)
> tensorboard --logdir=logs
