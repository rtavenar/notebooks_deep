"""
gan_trainer_skeleton_corr.py

Version corrigee du squelette "a trous" pour la seance 10 (modeles generatifs).

Contrairement au Trainer de training_toolbox.py (fourni complet), la boucle
d'entrainement alternee generateur / discriminateur (deux optimiseurs, mises
a jour non simultanees) ne rentre pas dans Trainer.fit telle quelle -- d'ou
cette classe a part. Les deux methodes _train_discriminator_step et
_train_generator_step, laissees a trous dans gan_trainer_skeleton.py, sont
ici completement implementees.

Usage attendu (une fois complete) :

    trainer = GANTrainer(generator, discriminator, opt_g, opt_d, latent_dim=64)
    history = trainer.fit(dataloader, epochs=20)
    samples = trainer.generate(16)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GANTrainer:
    def __init__(self, generator, discriminator, opt_g, opt_d, latent_dim,
                 device=None):
        self.generator = generator
        self.discriminator = discriminator
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.latent_dim = latent_dim
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.history = {"loss_g": [], "loss_d": []}

    def _sample_noise(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    # ------------------------------------------------------------------
    # TODO 1 : un pas d'entrainement du DISCRIMINATEUR.
    #
    #   1. Generer un batch d'images factices avec self.generator, SANS
    #      laisser le gradient remonter jusqu'au generateur (torch.no_grad()
    #      ou .detach() sur le resultat).
    #   2. Calculer self.loss_fn sur les vraies images (labels = 1) et sur
    #      les fausses (labels = 0), et sommer les deux.
    #   3. self.opt_d.zero_grad() puis backward() puis self.opt_d.step()
    #      -- ne pas toucher a self.opt_g ici.
    #   4. Retourner la valeur scalaire (float) de la loss.
    # ------------------------------------------------------------------
    def _train_discriminator_step(self, real_batch):
        batch_size = real_batch.size(0)

        with torch.no_grad():
            fake_batch = self.generator(self._sample_noise(batch_size))

        real_logits = self.discriminator(real_batch)
        fake_logits = self.discriminator(fake_batch)

        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        loss = self.loss_fn(real_logits, real_labels) + self.loss_fn(fake_logits, fake_labels)

        self.opt_d.zero_grad()
        loss.backward()
        self.opt_d.step()

        return loss.item()

    # ------------------------------------------------------------------
    # TODO 2 : un pas d'entrainement du GENERATEUR.
    #
    #   1. Generer un nouveau batch d'images factices -- cette fois le
    #      gradient DOIT pouvoir remonter jusqu'au generateur (pas de
    #      no_grad()/detach()).
    #   2. Les faire passer par self.discriminator.
    #   3. Loss du generateur : il cherche a tromper le discriminateur,
    #      donc on utilise le label 1 pour ces images pourtant fausses.
    #   4. self.opt_g.zero_grad() puis backward() puis self.opt_g.step()
    #      -- ne pas toucher a self.opt_d ici.
    #   5. Retourner la valeur scalaire (float) de la loss.
    # ------------------------------------------------------------------
    def _train_generator_step(self, batch_size):
        fake_batch = self.generator(self._sample_noise(batch_size))
        fake_logits = self.discriminator(fake_batch)

        real_labels = torch.ones(batch_size, 1, device=self.device)
        loss = self.loss_fn(fake_logits, real_labels)

        self.opt_g.zero_grad()
        loss.backward()
        self.opt_g.step()

        return loss.item()

    # -- le reste est deja ecrit ------------------------------------------
    def fit(self, dataloader, epochs=10, verbose=True):
        for epoch in range(epochs):
            running_g, running_d, n = 0.0, 0.0, 0
            for batch in dataloader:
                real_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
                real_batch = real_batch.to(self.device)
                bs = real_batch.size(0)

                loss_d = self._train_discriminator_step(real_batch)
                loss_g = self._train_generator_step(bs)

                running_d += loss_d * bs
                running_g += loss_g * bs
                n += bs

            self.history["loss_d"].append(running_d / n)
            self.history["loss_g"].append(running_g / n)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"loss_d: {running_d / n:.4f} - loss_g: {running_g / n:.4f}"
                )

        return self.history

    @torch.no_grad()
    def generate(self, n_samples=16):
        self.generator.eval()
        z = self._sample_noise(n_samples)
        samples = self.generator(z)
        self.generator.train()
        return samples
