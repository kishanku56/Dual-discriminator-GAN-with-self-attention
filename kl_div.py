import torch
import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import sys

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    checkpoint_epoch = sys.argv[1:][0] 
    file_path = f"all_fake_images_epoch_{checkpoint_epoch}.pt"
    fake_images = torch.load(file_path, map_location=device)  # [N,3,H,W]
    print(f"Loaded fake images: {fake_images.shape}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                            (0.5, 0.5, 0.5))  # normalize to [-1,1]
    ])
    dataset = dset.CIFAR10(root="./data", train=True, download=True, transform=transform)

    num_real = 10000
    real_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    real_images_list = []
    for imgs, _ in real_loader:
        real_images_list.append(imgs)
        if len(real_images_list) * 64 >= num_real:
            break
    real_images = torch.cat(real_images_list, dim=0)[:num_real]
    print(f"Loaded real CIFAR-10 subset: {real_images.shape}")


    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    inception = models.inception_v3(weights=weights).to(device)
    inception.eval()

    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    def get_probs(images):
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(images), 64):
                batch = images[i:i+64]
                # if images are in [-1,1], rescale to [0,1]
                if batch.min() < 0:
                    batch = (batch + 1) / 2
                # apply Inception preprocessing
                batch = torch.stack([preprocess(img.cpu()) for img in batch]).to(device)
                logits = inception(batch)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu())
        return torch.cat(all_probs, dim=0)

    print("Extracting Inception probabilities for real images...")
    probs_real = get_probs(real_images)
    print("Extracting Inception probabilities for fake images...")
    probs_fake = get_probs(fake_images)

    p_real = probs_real.mean(dim=0)
    p_fake = probs_fake.mean(dim=0)

    eps = 1e-12
    kl_real_fake = torch.sum(p_real * torch.log((p_real + eps) / (p_fake + eps)))
    kl_fake_real = torch.sum(p_fake * torch.log((p_fake + eps) / (p_real + eps)))

    print("\n===== KL Divergence Results =====")
    print(f"KL(real || fake): {kl_real_fake.item():.6f}")
    print(f"KL(fake || real): {kl_fake_real.item():.6f}")
