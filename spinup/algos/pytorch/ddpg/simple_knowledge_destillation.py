# Source: https://github.com/josehoras/Knowledge-Distillation/blob/master/knowledge_distillation.ipynb

import os
import torch
from pathlib import Path
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

xs = torch.linspace(-1.2, 0.6, steps=100)
ys = torch.linspace(-0.07, 0.07, steps=100)
x, y = torch.meshgrid(xs, ys, indexing='xy')

xr = torch.reshape(x, (-1, 1))
yr = torch.reshape(y, (-1, 1))
xyr = torch.stack((xr, yr), dim=1).squeeze()

# Source: https://stackoverflow.com/a/50098973
file_dir = str(Path().absolute())
pyt_save_path = os.path.join(file_dir, '../../../data/ddpg/ddpg_s0/pyt_save')

model_path = os.path.join(pyt_save_path, 'model_ac_pi_pi.pt')

ac_pi_pi_model = torch.load(model_path)

with torch.no_grad():
    z = ac_pi_pi_model(xyr)

full_dataset = TensorDataset(xyr, z)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

batch_size = 100
dl_train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
dl_test = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)


class StudentModelModule(LightningModule):
    def __init__(self, model, learning_rate=2e-4):
        super().__init__()

        # Set our init args as class attributes
        self.learning_rate = learning_rate

        self.model = model

        self.mseloss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x, y = batch
        out = self(x)

        loss = None

        try:
            loss = self.mseloss_fn(out, y)
        except:
            print(out, y)

        return out, loss

    def training_step(self, batch, batch_nb):
        out, loss = self._step(batch)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        out, loss = self._step(batch)

        self.log("test_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


trainer = Trainer(
    accelerator="gpu",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=50,
    callbacks=[TQDMProgressBar(refresh_rate=10)],
    logger=pl_loggers.TensorBoardLogger(save_dir="tensorboard_logs/"),
)

student_model = StudentModelModule(model, learning_rate=0.02)
trainer.fit(student_model, dl_train)

student_model_path = os.path.join(pyt_save_path, 'smaller_ac_pi_pi_model.pt')

torch.save(model, student_model_path)

trainer.test(student_model, dl_test)
