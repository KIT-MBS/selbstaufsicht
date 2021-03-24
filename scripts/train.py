import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer

from selbstaufsicht import models
from selbstaufsicht import datasets

class LitMod(LightningModule):

    def __init__(self):
        super().__init__()
        self.model = models.TransformerEncoderStack(4, 32, 4, 128, 12)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.crit(y, pred)

    def configure_optimizers(self):
        return Adam


class LitXfam(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_dims =None
        self.vocab_size = 4

    def prepare_data(self):
        download_dataset()

    def setup(self):
        self.train = load_train()

    def train_dataloader(self):
        transforms = None
        return DataLoader(self.train, batch_size=1)


model = LitMod()
trainer = Trainer()
trainer.fit(model, trainmod)



# nletters = 15


# def train(model, crit, train_loader):
#     model.train()
#     total_loss = 0.
#     start_time = time.time()
#     device = 'cpu'
#     model.to(device)
#     log_interval = 1
#     losses = []
#     # TODO fix data
#     # TODO try out lightning
#     # TODO fix devices
#     for i, data in enumerate(tqdm(train_loader, disable=False)):
#         optimizer.zero_grad()
#         x, y = data
#         x = x.transpose(0, 1) # l,
#         x.to(device)
#         y.to(device)
#         pred = model(x)
#
#         loss = crit(pred.view(-1, nletters), y.flatten())
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#         optimizer.step()
#
#         total_loss += loss.item()
#         if i % log_interval == 0 and i > 0:
#             cur_loss = total_loss / log_interval
#             elapsed = time.time() - start_time
#             # losses.append(total_loss)
#             losses.append(loss.item())
#             total_loss = 0.
#             start_time = time.time()
#     return losses
#
#
# if __name__ == "__main__":
#     model = models.TransformerEncoderStack(nletters, 32, 4, 128, 12)
#     crit = nn.CrossEntropyLoss()
#     lr = 5.
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1., gamma = 0.96)
#
#     train_loader = get_data_loader()
#
#     epochs = 5
#     losses = []
#     for epoch in range(1, epochs + 1):
#         losses.extend(train(model, crit, train_loader))
#         scheduler.step()
#
#     from matplotlib import pyplot as plt
#     plt.plot(losses)
#     plt.show()
#
#     print('fin')
