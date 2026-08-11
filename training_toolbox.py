"""
training_toolbox.py

Boite a outils d'entrainement PyTorch, pensee pour grossir seance apres
seance. Chaque section ci-dessous correspond a une etape de montee en
puissance : on demarre avec une boucle minimale (Trainer.fit) et on
n'ajoute JAMAIS de nouveau parametre obligatoire, uniquement des
parametres optionnels avec valeur par defaut -> le code des TPs deja
donnes aux etudiants continue de fonctionner sans modification.

Plan de montee en puissance (aligne sur la progression du cours,
11 seances) :
    Seance 1 (Prise en main PyTorch) -> pas de toolbox, torch pur
    Seance 2 (Optim manuelle)        -> pas de toolbox, torch pur
    Seance 3 (Optim structuree)      -> naissance du Trainer (fit/evaluate/
                                         plot)
    Seance 4 (Losses & init)         -> metriques (accuracy) suivies dans
                                         l'historique
    Seance 5 (Regularisation)        -> callbacks (EarlyStopping,
                                         ModelCheckpoint)
    Seance 6 (Mini-projet MLP)       -> consolidation, pas de nouvelle
                                         brique
    Seance 7-8 (CNN)                 -> consolidation (data augmentation
                                         cote torchvision.transforms, pas
                                         besoin de toucher au Trainer)
    Seance 9 (Mini-projet CNN)       -> consolidation, pas de nouvelle
                                         brique
    Seance 10 (Generatif)            -> consolidation, pas de nouvelle
                                         brique : autoencodeur et flow
                                         matching s'entrainent tous deux via
                                         Trainer.fit, a condition de bien
                                         choisir la paire (entree, cible)
                                         renvoyee par le Dataset
    Seance 11 (HuggingFace)          -> freeze/unfreeze, count_trainable_
                                         parameters, gestion des batches/
                                         models HF (deja presents ci-dessous)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch


# ---------------------------------------------------------------------------
# Seance 1 : la boucle d'entrainement minimale
# ---------------------------------------------------------------------------
class Trainer:
    """Encapsule la boucle d'entrainement / evaluation d'un modele torch.

    Usage minimal (identique du premier au dernier TP) :

        trainer = Trainer(model, optimizer, loss_fn)
        history = trainer.fit(train_loader, val_loader, epochs=10)
    """

    def __init__(self, model, optimizer, loss_fn, device=None,
                 metrics=None, callbacks=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device)

        # Seance 2 : {"acc": callable(preds, y) -> float}
        self.metrics = metrics or {}
        # Seance 3 : liste de Callback
        self.callbacks = callbacks or []

        self.history = {"train_loss": [], "val_loss": []}
        for name in self.metrics:
            self.history[f"train_{name}"] = []
            self.history[f"val_{name}"] = []

    # -- points d'extension pour gerer differents types de batches ---------
    def _prepare_batch(self, batch):
        """(x, y) classique OU batch HuggingFace (dict avec 'labels').
        Seance 5 : c'est ici qu'on branche les modeles pre-entraines."""
        if isinstance(batch, dict):
            batch = dict(batch)
            labels = batch.pop("labels")
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            return inputs, labels.to(self.device)
        x, y = batch
        return x.to(self.device), y.to(self.device)

    def _forward(self, inputs):
        """Uniformise l'appel modele "classique" (retourne un tenseur) et
        modele HuggingFace (retourne un objet avec .logits)."""
        if isinstance(inputs, dict):
            out = self.model(**inputs)
            return out.logits if hasattr(out, "logits") else out
        return self.model(inputs)

    # -- coeur de la boucle --------------------------------------------------
    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        totals = {"loss": 0.0, **{name: 0.0 for name in self.metrics}}
        n = 0
        with torch.set_grad_enabled(train):
            for batch in loader:
                inputs, y = self._prepare_batch(batch)
                preds = self._forward(inputs)
                loss = self.loss_fn(preds, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                bs = y.size(0)
                totals["loss"] += loss.item() * bs
                for name, fn in self.metrics.items():
                    totals[name] += fn(preds.detach(), y).item() * bs
                n += bs

        return {k: v / n for k, v in totals.items()}

    def fit(self, train_loader, val_loader=None, epochs=10, verbose=True):
        for epoch in range(epochs):
            train_stats = self._run_epoch(train_loader, train=True)
            logs = {"epoch": epoch}
            for k, v in train_stats.items():
                key = "train_loss" if k == "loss" else f"train_{k}"
                self.history[key].append(v)
                logs[key] = v

            if val_loader is not None:
                val_stats = self._run_epoch(val_loader, train=False)
                for k, v in val_stats.items():
                    key = "val_loss" if k == "loss" else f"val_{k}"
                    self.history[key].append(v)
                    logs[key] = v

            for cb in self.callbacks:
                cb.on_epoch_end(self, logs)

            if verbose:
                parts = [f"{k}: {v:.4f}" for k, v in logs.items() if k != "epoch"]
                print(f"Epoch {epoch + 1}/{epochs} - " + " - ".join(parts))

            if any(getattr(cb, "stop_training", False) for cb in self.callbacks):
                if verbose:
                    print("Early stopping.")
                break

        return self.history

    def evaluate(self, loader):
        return self._run_epoch(loader, train=False)

    def plot(self, metrics=None, figsize=None):
        """Trace loss (et les metriques suivies) train vs val en fonction des
        epochs. Un sous-graphe par quantite trace ; utilisable des qu'un
        `fit` a ete appele.

            trainer.plot()               # loss + toutes les metriques suivies
            trainer.plot(metrics=["acc"])  # uniquement loss + acc
        """
        names = ["loss"] + list(self.metrics if metrics is None else metrics)
        figsize = figsize or (5 * len(names), 4)
        fig, axes = plt.subplots(1, len(names), figsize=figsize)
        axes = [axes] if len(names) == 1 else axes
        for ax, name in zip(axes, names):
            ax.plot(self.history[f"train_{name}"], label="train")
            if self.history.get(f"val_{name}"):
                ax.plot(self.history[f"val_{name}"], label="val")
            ax.set_xlabel("epoch")
            ax.set_title(name)
            ax.legend()
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Seance 3 : callbacks
# ---------------------------------------------------------------------------
class Callback:
    """Classe de base. Redefinir on_epoch_end (et, plus tard, on_train_begin,
    on_batch_end...) dans des sous-classes."""
    stop_training = False

    def on_epoch_end(self, trainer, logs):
        pass


class EarlyStopping(Callback):
    def __init__(self, patience=5, monitor="val_loss"):
        self.patience = patience
        self.monitor = monitor
        self.best = float("inf")
        self.wait = 0
        self.stop_training = False

    def on_epoch_end(self, trainer, logs):
        value = logs.get(self.monitor)
        if value is None:
            return
        if value < self.best:
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
            self.stop_training = self.wait >= self.patience


class ModelCheckpoint(Callback):
    def __init__(self, path, monitor="val_loss"):
        self.path = path
        self.monitor = monitor
        self.best = float("inf")

    def on_epoch_end(self, trainer, logs):
        value = logs.get(self.monitor)
        if value is not None and value < self.best:
            self.best = value
            torch.save(trainer.model.state_dict(), self.path)


class PeriodicCheckpoint(Callback):
    def __init__(self, path_template="model_epoch{epoch}.pt", every=10):
        self.path_template = path_template
        self.every = every

    def on_epoch_end(self, trainer, logs):
        epoch = logs["epoch"] + 1  # logs["epoch"] est l'indice de l'epoch, 0-based
        if epoch % self.every == 0:
            torch.save(trainer.model.state_dict(), self.path_template.format(epoch=epoch))


# ---------------------------------------------------------------------------
# Seance 4 : utilitaires pour le transfer learning (CNN, puis HF)
# ---------------------------------------------------------------------------
def freeze(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Seance 2 (metrique fournie clef en main pour la classification)
# ---------------------------------------------------------------------------
def accuracy(preds, y):
    return (preds.argmax(dim=-1) == y).float().mean()

