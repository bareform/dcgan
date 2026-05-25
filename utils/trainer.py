from models import (
    Generator,
    Discriminator
)

import os

from cleanfid import fid
from datasets import load_dataset
from PIL import Image
import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import torchutils

from .hf_utils import (
    ImageDataset
)
from .gen_utils import (
    weights_init,
    gen,
)

def get_argparser():
    parser = torchutils.ArgumentParser("Simple training loop.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["afhq-cat", "afhq-wild"],
        help="Dataset to train on. Must be one of: `afhq-cat`, `afhq-wild`.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads to use in data loading (default: 4)"
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=True,
        help="Whether to pin memory for data loading (default: True)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Number of epochs to train for (default: 200)",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=64,
        help="Height of the input images (default: 64)",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=64,
        help="Width of the input images (default: 64)",
    )
    parser.add_argument(
        "--generator_lr",
        type=float,
        default=0.001,
        help="Learning rate for the generator (default: 0.001).",
    )
    parser.add_argument(
        "--discriminator_lr",
        type=float,
        default=0.001,
        help="Learning rate for the discriminator (default: 0.001).",
    )
    parser.add_argument(
        "--generator_adam_beta1",
        type=float,
        default=0.5,
        help="Adam beta1 (default: 0.5).",
    )
    parser.add_argument(
        "--generator_adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2 (default: 0.999).",
    )
    parser.add_argument(
        "--discriminator_adam_beta1",
        type=float,
        default=0.5,
        help="Adam beta1 (default: 0.5).",
    )
    parser.add_argument(
        "--discriminator_adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2 (default: 0.999).",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=128,
        help="Latent space dimensionality (default: 128).",
    )
    parser.add_argument(
        "--generator_in_channels",
        type=int,
        nargs="+",
        help="Input channels for the generator.",
    )
    parser.add_argument(
        "--discriminator_in_channels",
        type=int,
        nargs="+",
        help="Input channels for the discriminator.",
    )
    parser.add_argument(
        "--use_spectral_norm",
        action="store_true",
        help="Whether to use spectral normalization (default: False).",
    )
    parser.add_argument(
        "--use_hinge_loss",
        action="store_true",
        help="Whether to use hinge loss (default: False).",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=os.path.join(".", "checkpoints"),
        help="Directory to save checkpoints (default: ./checkpoints).",
    )
    parser.add_argument(
        "--save_ckpt_interval",
        type=int,
        default=200,
        help="Interval (in epochs) at which to save model checkpoints (default: 200).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(".", "output"),
        help="Directory to save intermediate output (default: ./output).",
    )
    parser.add_argument(
        "--save_output_interval",
        type=int,
        default=50,
        help="Number of epochs between saving intermediate output (default: 50).",
    )
    parser.add_argument(
        "--nrow",
        type=int,
        default=9,
        help="Number of rows in output grid (default: 9).",
    )
    parser.add_argument(
        "--compute_fid_interval",
        type=int,
        default=50,
        help="Number of epochs between FID evaluations (default: 50).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    return parser

def main():
    args = get_argparser().parse_args()
    pad_length = len(str(args.num_epochs))

    torchutils.set_seed(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Training on {args.dataset}")
    if args.dataset == "afhq-cat":
        hf_dataset = load_dataset("luethan2025/AFHQ-Cat-64x64", split="train")
        transform = transforms.Compose([
            transforms.Resize(args.image_height),
            transforms.CenterCrop(args.image_height),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = ImageDataset(
            hf_dataset=hf_dataset,
            transform=transform
        )
        dataloader = data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
        dataset = "afhq-cat-64x64"
        img_size = (3, args.image_height, args.image_width)
    elif args.dataset == "afhq-wild":
        hf_dataset = load_dataset("luethan2025/AFHQ-Wild-64x64", split="train")
        transform = transforms.Compose([
            transforms.Resize(args.image_height),
            transforms.CenterCrop(args.image_height),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = ImageDataset(
            hf_dataset=hf_dataset,
            transform=transform
        )
        dataloader = data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
        dataset = "afhq-wild-64x64"
        img_size = (3, args.image_height, args.image_width)

    data_dir = os.path.join("dcgan_data", dataset)
    os.makedirs(data_dir, exist_ok=True)
    file_pad = len(str(len(hf_dataset)))
    for idx, item in enumerate(hf_dataset):
        image = item['image']
        image.save(os.path.join(data_dir, f"{str(idx).zfill(file_pad)}.png"))

    if not fid.test_stats_exists(dataset, mode="clean"):
        fid.make_custom_stats(dataset, data_dir, mode="clean")

    G = Generator(
        img_size=img_size,
        in_channels=args.generator_in_channels,
        latent_dim=args.latent_dim,
    )
    G.apply(weights_init)
    D = Discriminator(
        img_size=img_size,
        in_channels=args.discriminator_in_channels,
        use_spectral_norm=args.use_spectral_norm
    )
    if not args.use_spectral_norm:
        D.apply(weights_init)
    G = G.to(device)
    D = D.to(device)

    G_optimizer = optim.Adam(G.parameters(), lr=args.generator_lr, betas=(args.generator_adam_beta1, args.generator_adam_beta2))
    D_optimizer = optim.Adam(D.parameters(), lr=args.discriminator_lr, betas=(args.discriminator_adam_beta1, args.discriminator_adam_beta2))
    criterion = torch.nn.BCEWithLogitsLoss()

    if not (os.path.exists(args.ckpt_dir) and os.path.isdir(args.ckpt_dir)):
        os.makedirs(args.ckpt_dir, exist_ok=True)

    output_dir = os.path.join(args.output_dir, args.dataset, "epoch")
    if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
        os.makedirs(output_dir, exist_ok=True)

    G.train()
    D.train()
    test_noise = torch.randn(args.nrow ** 2, args.latent_dim, 1, 1, device=device)

    real_label_value = 1.0
    fake_label_value = 0.0
    for epoch in range(args.num_epochs):
        with tqdm(dataloader, desc=f"Training", unit="batch") as pbar:
            running_G_loss = 0.0
            running_D_loss = 0.0
            running_real_loss = 0.0
            running_fake_loss = 0.0
            for idx, images in enumerate(pbar):
                images = images.to(device)
                batch_size = images.size(0)

                real_labels = torch.full((batch_size,), real_label_value, device=device)
                fake_labels = torch.full((batch_size,), fake_label_value, device=device)

                # === Train Discriminator ===
                D_optimizer.zero_grad()
                noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
                fake_images = G(noise)
                real_out = D(images)
                fake_out = D(fake_images.detach())

                if args.use_hinge_loss:
                    real_loss = torch.mean(torch.clamp(1.0 - real_out, min=0.0))
                    fake_loss = torch.mean(torch.clamp(1.0 + fake_out, min=0.0))
                else:
                    real_loss = criterion(real_out.view(-1), real_labels)
                    fake_loss = criterion(fake_out.view(-1), fake_labels)

                D_loss = real_loss + fake_loss
                D_loss.backward()
                D_optimizer.step()

                running_D_loss += D_loss.item()
                running_real_loss += real_loss.item()
                running_fake_loss += fake_loss.item()

                # === Train Generator ===
                G_optimizer.zero_grad()

                noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
                fake_images = G(noise)

                if args.use_hinge_loss:
                    G_loss = -torch.mean(D(fake_images))
                else:
                    G_loss = criterion(D(fake_images).view(-1), real_labels)

                G_loss.backward()
                G_optimizer.step()

                running_G_loss += G_loss.item()

                pbar.set_postfix({
                    "G_loss": f"{G_loss.item():.2f}",
                    "D_loss": f"{D_loss.item():.2f}",
                    "D(real)": f"{real_loss.item():.2f}",
                    "D(fake)": f"{fake_loss.item():.2f}"
                })
        average_G_loss = running_G_loss / len(dataloader)
        average_D_loss = running_D_loss / len(dataloader)
        average_real_loss = running_real_loss / len(dataloader)
        average_fake_loss = running_fake_loss / len(dataloader)
        print(f"Epoch: {epoch + 1}/{args.num_epochs}")
        print(f"G Loss: {average_G_loss:.5f}, D Loss: {average_D_loss:.5f}, D(real): {average_real_loss:.5f}, D(fake): {average_fake_loss:.5f}")

        if (epoch + 1) % args.save_output_interval == 0:
            print("Saving fake images")
            G.eval()
            with torch.no_grad():
                fake_images = G(test_noise)
                grid = torchvision.utils.make_grid(fake_images, nrow=args.nrow, normalize=True)
                torchvision.utils.save_image(
                    grid,
                    os.path.join(output_dir, f"{epoch + 1:0{pad_length}d}.png")
                )
            G.train()

        if (epoch + 1) % args.save_ckpt_interval == 0:
            print("Saving model checkpoints")
            checkpoint = {
                "dataset": args.dataset,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "generator_in_channels": args.generator_in_channels,
                "discriminator_in_channels": args.discriminator_in_channels,
                "use_spectral_norm": args.use_spectral_norm,
                "use_hinge_loss": args.use_hinge_loss,
                "latent_dim": args.latent_dim,
                "generator_lr": args.generator_lr,
                "discriminator_lr": args.discriminator_lr,
                "G_optimizer": G_optimizer.state_dict(),
                "D_optimizer": D_optimizer.state_dict(),
                "image_height": args.image_height,
                "image_width": args.image_width
            }
            torch.save(checkpoint, os.path.join(args.ckpt_dir, f"{args.dataset}_checkpoint_{epoch + 1:0{pad_length}d}.pth"))
            generator = {
                "dataset": args.dataset,
                "G": G.state_dict(),
                "generator_in_channels": args.generator_in_channels,
                "latent_dim": args.latent_dim,
                "img_size": img_size,
            }
            torch.save(generator, os.path.join(args.ckpt_dir, f"{args.dataset}_{epoch + 1:0{pad_length}d}.pth"))

        if (epoch + 1) % args.compute_fid_interval == 0:
            fid_score = fid.compute_fid(
                gen=lambda z: gen(G, z, device),
                dataset_name=dataset,
                dataset_res=64,
                num_gen=10_000,
                dataset_split="custom"
            )
            print(f"FID: {fid_score}")

    gif_dir = os.path.join(args.output_dir, args.dataset, "gif")
    if not (os.path.exists(gif_dir) and os.path.isdir(gif_dir)):
        os.makedirs(gif_dir, exist_ok=True)

    print("Saving gif")
    output_gif = os.path.join(gif_dir, f"{args.dataset}_{epoch + 1:0{pad_length}d}.gif")
    png_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])
    images = [Image.open(os.path.join(output_dir, f)) for f in png_files]
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=150,
        loop=0
    )

    print("Saving final fake images")
    G.eval()
    with torch.no_grad():
        fake_images = G(test_noise)
        grid = torchvision.utils.make_grid(fake_images, nrow=args.nrow, normalize=True)
        torchvision.utils.save_image(
            grid,
            os.path.join(output_dir, f"{epoch + 1:0{pad_length}d}.png")
        )
    G.train()

    print("Saving final model checkpoints")
    checkpoint = {
        "dataset": args.dataset,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "generator_in_channels": args.generator_in_channels,
        "discriminator_in_channels": args.discriminator_in_channels,
        "use_spectral_norm": args.use_spectral_norm,
        "use_hinge_loss": args.use_hinge_loss,
        "latent_dim": args.latent_dim,
        "generator_lr": args.generator_lr,
        "discriminator_lr": args.discriminator_lr,
        "G_optimizer": G_optimizer.state_dict(),
        "D_optimizer": D_optimizer.state_dict(),
        "image_height": args.image_height,
        "image_width": args.image_width
    }
    torch.save(checkpoint, os.path.join(args.ckpt_dir, f"{args.dataset}_checkpoint_{epoch + 1:0{pad_length}d}.pth"))
    generator = {
        "dataset": args.dataset,
        "G": G.state_dict(),
        "generator_in_channels": args.generator_in_channels,
        "latent_dim": args.latent_dim,
        "img_size": img_size,
    }
    torch.save(generator, os.path.join(args.ckpt_dir, f"{args.dataset}_{epoch + 1:0{pad_length}d}.pth"))

if __name__ == "__main__":
    main()
