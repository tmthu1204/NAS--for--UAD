# # src/trainers/ts_trainer.py
# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from src.models.loss import NTXentLoss


# class TSTrainer:
#     def __init__(self, model, temporal_contr_model, model_optimizer, temp_cont_optimizer, device, config, logger=None):
#         self.model = model
#         self.temporal_contr_model = temporal_contr_model
#         self.model_optimizer = model_optimizer
#         self.temp_cont_optimizer = temp_cont_optimizer
#         self.device = device
#         self.config = config
#         self.logger = logger or print

#         self.criterion = nn.CrossEntropyLoss()
#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.model_optimizer, 'min')

#     def train(self, train_dl, valid_dl=None, test_dl=None, training_mode="self_supervised", experiment_log_dir="results"):
#         self.logger(">>> Training started ...")

#         for epoch in range(1, self.config.num_epoch + 1):
#             train_loss, train_acc = self._train_one_epoch(train_dl, training_mode)
#             valid_loss, valid_acc, _, _ = self.evaluate(valid_dl, training_mode)

#             if training_mode != 'self_supervised':
#                 self.scheduler.step(valid_loss)

#             self.logger(f"[Epoch {epoch}] Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
#                         f"Valid Loss={valid_loss:.4f}, Valid Acc={valid_acc:.4f}")

#         # save checkpoint
#         os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
#         chkpoint = {
#             'model_state_dict': self.model.state_dict(),
#             'temporal_contr_model_state_dict': self.temporal_contr_model.state_dict()
#         }
#         torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", "ckp_last.pt"))

#         if training_mode != "self_supervised" and test_dl is not None:
#             self.logger(">>> Evaluating on Test set ...")
#             test_loss, test_acc, _, _ = self.evaluate(test_dl, training_mode)
#             self.logger(f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

#         self.logger(">>> Training finished.")

#     def _train_one_epoch(self, train_loader, training_mode):
#         total_loss, total_acc = [], []
#         self.model.train()
#         self.temporal_contr_model.train()

#         # ✅ Debug batch structure
#         for batch in train_loader:
#             print("[DEBUG] First batch type:", type(batch))
#             if isinstance(batch, (tuple, list)):
#                 print("[DEBUG] Batch length:", len(batch))
#                 for i, b in enumerate(batch):
#                     if isinstance(b, torch.Tensor):
#                         print(f"[DEBUG] batch[{i}] shape:", b.shape)
#                     else:
#                         print(f"[DEBUG] batch[{i}] type:", type(b))
#             else:
#                 print("[DEBUG] Batch is not tuple/list:", batch)
#             break  # chỉ in batch đầu tiên thôi

#         for data, labels, aug1, aug2 in train_loader:
#             data, labels = data.float().to(self.device), labels.long().to(self.device)
#             aug1, aug2 = aug1.float().to(self.device), aug2.float().to(self.device)

#             self.model_optimizer.zero_grad()
#             self.temp_cont_optimizer.zero_grad()

#             if training_mode == "self_supervised":
#                 predictions1, features1 = self.model(aug1)
#                 predictions2, features2 = self.model(aug2)

#                 features1 = F.normalize(features1, dim=1)
#                 features2 = F.normalize(features2, dim=1)

#                 temp_cont_loss1, feat1 = self.temporal_contr_model(features1, features2)
#                 temp_cont_loss2, feat2 = self.temporal_contr_model(features2, features1)

#                 zis, zjs = feat1, feat2

#                 nt_xent = NTXentLoss(self.device, self.config.batch_size,
#                                      self.config.Context_Cont.temperature,
#                                      self.config.Context_Cont.use_cosine_similarity)
#                 loss = (temp_cont_loss1 + temp_cont_loss2) + 0.7 * nt_xent(zis, zjs)

#             else:  # supervised / fine-tune
#                 predictions, features = self.model(data)
#                 loss = self.criterion(predictions, labels)
#                 total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

#             total_loss.append(loss.item())
#             loss.backward()
#             self.model_optimizer.step()
#             self.temp_cont_optimizer.step()

#         avg_loss = torch.tensor(total_loss).mean().item()
#         avg_acc = torch.tensor(total_acc).mean().item() if total_acc else 0.0
#         return avg_loss, avg_acc

#     def evaluate(self, dataloader, training_mode):
#         if dataloader is None:
#             return 0, 0, [], []
#         self.model.eval()
#         self.temporal_contr_model.eval()

#         total_loss, total_acc = [], []
#         outs, trgs = np.array([]), np.array([])

#         with torch.no_grad():
#             for data, labels, _, _ in dataloader:
#                 data, labels = data.float().to(self.device), labels.long().to(self.device)

#                 if training_mode != "self_supervised":
#                     predictions, features = self.model(data)
#                     loss = self.criterion(predictions, labels)
#                     total_acc.append(labels.eq(predictions.argmax(dim=1)).float().mean())
#                     total_loss.append(loss.item())

#                     pred = predictions.argmax(dim=1, keepdim=True)
#                     outs = np.append(outs, pred.cpu().numpy())
#                     trgs = np.append(trgs, labels.cpu().numpy())

#         avg_loss = torch.tensor(total_loss).mean().item() if total_loss else 0.0
#         avg_acc = torch.tensor(total_acc).mean().item() if total_acc else 0.0
#         return avg_loss, avg_acc, outs, trgs


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.ts_tcc.models.loss import NTXentLoss


class TSTrainer:
    def __init__(self, model, temporal_contr_model, model_optimizer, temp_cont_optimizer, device, config, logger=None):
        self.model = model
        self.temporal_contr_model = temporal_contr_model
        self.model_optimizer = model_optimizer
        self.temp_cont_optimizer = temp_cont_optimizer
        self.device = device
        self.config = config
        self.logger = logger or print

        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.model_optimizer, 'min')

        # Reuse NTXentLoss instance
        self.nt_xent = NTXentLoss(
            self.device,
            self.config.batch_size,
            self.config.Context_Cont.temperature,
            self.config.Context_Cont.use_cosine_similarity
        )

        # Paper coefficients
        self.lambda1 = 1.0
        self.lambda2 = 0.7

    def train(self, train_dl, valid_dl=None, test_dl=None, training_mode="self_supervised", experiment_log_dir="results"):
        self.logger(">>> Training TS-TCC ...")

        for epoch in range(1, self.config.num_epoch + 1):
            train_loss, train_acc = self._train_one_epoch(train_dl, training_mode)

            # Only evaluate & step scheduler when not self-supervised
            if training_mode != "self_supervised" and valid_dl is not None:
                valid_loss, valid_acc, _, _ = self.evaluate(valid_dl, training_mode)
                self.scheduler.step(valid_loss)
            else:
                valid_loss, valid_acc = 0.0, 0.0

            self.logger(
                f"[Epoch {epoch}] "
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}"
                + (f", Valid Loss={valid_loss:.4f}, Valid Acc={valid_acc:.4f}" if training_mode != "self_supervised" else "")
            )

        # === Save checkpoint ===
        os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
        chkpoint = {
            "model_state_dict": self.model.state_dict(),
            "temporal_contr_model_state_dict": self.temporal_contr_model.state_dict()
        }
        torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", "ckp_last.pt"))

        # === Evaluate (optional) ===
        if training_mode != "self_supervised" and test_dl is not None:
            self.logger(">>> Evaluating on Test set ...")
            test_loss, test_acc, _, _ = self.evaluate(test_dl, training_mode)
            self.logger(f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

        self.logger(">>> TS-TCC training finished.\n")

    def _train_one_epoch(self, train_loader, training_mode):
        total_loss, total_acc = [], []
        self.model.train()
        self.temporal_contr_model.train()

        for data, labels, aug1, aug2 in train_loader:
            data, labels = data.float().to(self.device), labels.long().to(self.device)
            aug1, aug2 = aug1.float().to(self.device), aug2.float().to(self.device)

            self.model_optimizer.zero_grad()
            self.temp_cont_optimizer.zero_grad()

            if training_mode == "self_supervised":
                # === Forward passes for two augmentations ===
                _, feat1 = self.model(aug1)
                _, feat2 = self.model(aug2)

                # Normalize features
                feat1 = F.normalize(feat1, dim=1)
                feat2 = F.normalize(feat2, dim=1)

                # Temporal contrasting (cross-view prediction)
                temp_loss1, temp_feat1 = self.temporal_contr_model(feat1, feat2)
                temp_loss2, temp_feat2 = self.temporal_contr_model(feat2, feat1)

                # Contextual contrasting
                zis = F.normalize(temp_feat1, dim=1)
                zjs = F.normalize(temp_feat2, dim=1)

                loss = (temp_loss1 + temp_loss2) * self.lambda1 + self.lambda2 * self.nt_xent(zis, zjs)

            else:  # supervised / fine-tune
                predictions, _ = self.model(data)
                loss = self.criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

            total_loss.append(loss.item())
            loss.backward()
            self.model_optimizer.step()
            self.temp_cont_optimizer.step()

        avg_loss = torch.tensor(total_loss).mean().item()
        avg_acc = torch.tensor(total_acc).mean().item() if total_acc else 0.0
        return avg_loss, avg_acc

    def evaluate(self, dataloader, training_mode):
        if dataloader is None:
            return 0.0, 0.0, [], []

        # In self-supervised mode we skip evaluation entirely
        if training_mode == "self_supervised":
            return 0.0, 0.0, [], []

        self.model.eval()
        self.temporal_contr_model.eval()

        total_loss, total_acc = [], []
        outs, trgs = np.array([]), np.array([])
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, labels, _, _ in dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                preds, _ = self.model(data)
                loss = criterion(preds, labels)
                total_acc.append(labels.eq(preds.argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

                outs = np.append(outs, preds.argmax(dim=1).cpu().numpy())
                trgs = np.append(trgs, labels.cpu().numpy())

        avg_loss = torch.tensor(total_loss).mean().item() if total_loss else 0.0
        avg_acc = torch.tensor(total_acc).mean().item() if total_acc else 0.0
        return avg_loss, avg_acc, outs, trgs
