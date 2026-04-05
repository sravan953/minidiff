import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
from minidiff.unet import UNet

PATH_MNIST = str(Path(__file__).parent.parent)
SAMPLES_FOLDER = "output/samples"


def train(
    num_epochs: int,
    batch_size: int,
    lr: float,
    num_diffusion_steps: int,
    val_every_n_epochs: int,
):
    # -------------------- Datasets and dataloaders --------------------
    train_transforms = T.Compose(
        [
            T.PILToTensor(),
            T.ToDtype(torch.float32, scale=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.Lambda(lambda x: (x - x.mean()) / x.std()),
        ]
    )
    train_mnist = torchvision.datasets.MNIST(
        PATH_MNIST, train=True, download=True, transform=train_transforms
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_mnist,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )
    num_train_batches = len(train_dataloader.dataset) // batch_size

    val_transforms = T.Compose(
        [
            T.PILToTensor(),
            T.ToDtype(torch.float32, scale=True),
            T.Lambda(lambda x: (x - x.mean()) / x.std()),
        ]
    )
    val_mnist = torchvision.datasets.MNIST(
        PATH_MNIST, train=False, download=True, transform=val_transforms
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_mnist,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )
    num_val_batches = len(val_dataloader.dataset) // batch_size

    # -------------------- DDPM Setup --------------------
    beta_start = 1e-4
    beta_end = 0.02
    beta_schedule = torch.linspace(
        beta_start, beta_end, num_diffusion_steps, device="cuda"
    )
    alpha_schedule = 1.0 - beta_schedule
    alpha_cum_prod = torch.cumprod(alpha_schedule, dim=0).view(-1, 1, 1, 1)
    sqrt_alpha_cum_prod = torch.sqrt(alpha_cum_prod)
    sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - alpha_cum_prod)

    # -------------------- Training loop --------------------
    model = UNet(num_blocks=4).cuda()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-5,
    )

    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        # -------------------- Backprop --------------------
        model.train()
        print(f"[Training  ] Epoch {epoch + 1}/{num_epochs}", end="\r")
        train_epoch_losses = []
        for i, (x_0, _) in enumerate(train_dataloader):
            x_0 = x_0.cuda()
            timesteps = torch.randint(
                1, num_diffusion_steps + 1, (x_0.shape[0],), device="cuda"
            )
            noise = torch.randn_like(x_0)
            x_noisy = (
                sqrt_alpha_cum_prod[timesteps - 1] * x_0
                + sqrt_one_minus_alpha_cum_prod[timesteps - 1] * noise
            )
            noise_pred = model(x_noisy, timesteps)
            loss = nn.functional.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_losses.append(loss.item())
            print(
                f"[Training  ] Epoch {epoch + 1}/{num_epochs} | Step {i + 1}/{num_train_batches} | Loss: {loss.item():.4g}",
                end="\r",
            )
        train_losses.append(train_epoch_losses)
        print(
            f"[Training  ] Epoch {epoch + 1}/{num_epochs} | Step {i + 1}/{num_train_batches} | Loss: {sum(train_epoch_losses) / len(train_epoch_losses):.4g}"
        )

        # -------------------- Validation --------------------
        if (epoch + 1) % val_every_n_epochs == 0:
            model.eval()
            val_epoch_losses = []
            with torch.no_grad():
                for i, (x_0, _) in enumerate(val_dataloader):
                    x_0 = x_0.cuda()
                    timesteps = torch.randint(
                        1, num_diffusion_steps + 1, (x_0.shape[0],), device="cuda"
                    )
                    noise = torch.randn_like(x_0)
                    if i == 0:
                        # -------------------- Generate samples --------------------
                        num_samples = 6
                        x_t = torch.randn_like(x_0[:num_samples])
                        for ts in range(num_diffusion_steps, 0, -1):
                            ts_t = torch.tensor([ts] * num_samples, device="cuda")
                            z_t = torch.randn_like(x_t) if ts > 1 else 0
                            noise_pred = model(x_t, ts_t)
                            term1 = 1 / torch.sqrt(alpha_schedule[ts - 1])
                            term2 = (
                                (1 - alpha_schedule[ts - 1])
                                / torch.sqrt(1 - alpha_cum_prod[ts - 1].squeeze())
                            ).reshape(-1, 1, 1, 1)
                            term3 = (
                                torch.sqrt(beta_schedule[ts - 1]).reshape(-1, 1, 1, 1)
                                * z_t
                            )
                            x_t_minus_1 = (term1 * (x_t - (term2 * noise_pred))) + term3
                            x_t = x_t_minus_1
                        x_t_minus_1 = x_t_minus_1.reshape(num_samples, 28, 28)
                        lo = torch.amin(x_t_minus_1, dim=(1, 2), keepdim=True)
                        hi = torch.amax(x_t_minus_1, dim=(1, 2), keepdim=True)
                        x_t_minus_1 = (x_t_minus_1 - lo) / (hi - lo + 1e-6)
                        x_t_minus_1 = torch.cat([x_i for x_i in x_t_minus_1], dim=1)
                        x_t_minus_1 = x_t_minus_1.cpu().numpy()
                        plt.imshow(x_t_minus_1, cmap="gray")
                        plt.title(f"Epoch {epoch + 1}")
                        plt.axis("off")
                        plt.savefig(
                            f"{SAMPLES_FOLDER}/epoch_{epoch + 1}.png",
                            bbox_inches="tight",
                        )
                        plt.close()
                        # -----------------------------------------------------------

                    x_noisy = (
                        sqrt_alpha_cum_prod[timesteps - 1] * x_0
                        + sqrt_one_minus_alpha_cum_prod[timesteps - 1] * noise
                    )
                    noise_pred = model(x_noisy, timesteps)
                    loss = nn.functional.mse_loss(noise_pred, noise)
                    val_epoch_losses.append(loss.item())
                    print(
                        f"[Validation] Epoch {epoch + 1}/{num_epochs} | Step {i + 1}/{num_val_batches} | Loss: {loss.item():.4g}",
                        end="\r",
                    )
                mean_val_loss = sum(val_epoch_losses) / len(val_epoch_losses)
                # -------------------- Save best model --------------------
                save_suffix = ""
                if mean_val_loss < min(val_losses, default=float("inf")):
                    torch.save(model.state_dict(), f"best_model.ckpt")
                    save_suffix = " | Current best loss, saving model"
                print(
                    f"[Validation] Epoch {epoch + 1}/{num_epochs} | Step {i + 1}/{num_val_batches} | Loss: {mean_val_loss:.4g}"
                    + save_suffix
                )
            val_losses.append(mean_val_loss)


if __name__ == "__main__":
    num_epochs = 1000
    batch_size = 1024
    lr = 2e-4
    num_diffusion_steps = 1000
    val_every_n_epochs = 5
    Path(SAMPLES_FOLDER).mkdir(exist_ok=True)
    train(num_epochs, batch_size, lr, num_diffusion_steps, val_every_n_epochs)
