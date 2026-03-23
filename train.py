import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import mlflow
import mlflow.pytorch

# Use env var for tracking URI (set via GitHub Actions secret)
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)

# SETUP MLFLOW
mlflow.set_experiment("Assignment3_Rahma")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 100
batch_size = 64
lr = 0.0002
epochs = 2
img_size = 28


# Define Models
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(True),
            nn.Linear(256, 512), nn.ReLU(True),
            nn.Linear(512, 1024), nn.ReLU(True),
            nn.Linear(1024, img_size * img_size), nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z).view(z.size(0), 1, img_size, img_size)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size, 512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img.view(img.size(0), -1))


# Data Setup
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# MLFLOW RUN
with mlflow.start_run() as run:
    # Log Tags & Hyperparameters
    mlflow.set_tag("student_id", "202202059")
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("optimizer", "Adam")

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    final_g_loss = None

    for epoch in range(epochs):
        running_d_loss = 0.0
        running_g_loss = 0.0

        for i, (imgs, _) in enumerate(dataloader):
            real_imgs = imgs.to(device)
            batch_size_curr = real_imgs.size(0)
            valid = torch.ones(batch_size_curr, 1, device=device)
            fake = torch.zeros(batch_size_curr, 1, device=device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size_curr, latent_dim, device=device)
            gen_imgs = generator(z)
            g_loss = criterion(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(real_imgs), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()

        # LIVE LOGGING (End of Epoch)
        avg_d_loss = running_d_loss / len(dataloader)
        avg_g_loss = running_g_loss / len(dataloader)
        final_g_loss = avg_g_loss

        mlflow.log_metric("d_loss", avg_d_loss, step=epoch)
        mlflow.log_metric("g_loss", avg_g_loss, step=epoch)

        print(f"Epoch {epoch} | D-Loss: {avg_d_loss:.4f} | G-Loss: {avg_g_loss:.4f}")

    # Log final summary metrics for threshold check
    mlflow.log_metric("final_g_loss", final_g_loss)

    # Save Model Artifact
    mlflow.pytorch.log_model(generator, "generator_model")

    # ── EXPORT run ID so the deploy job can find this run ──────────────────
    run_id = run.info.run_id
    with open("model_info.txt", "w") as f:
        f.write(run_id)

    print(f"\nRun Complete. Run ID: {run_id}")
    print(f"Final G-Loss: {final_g_loss:.4f}")
    print("model_info.txt written. Check MLflow UI.")
