import os
import pytorch_lightning as pl
from datetime import datetime
from pathlib import Path


class WMCustomModelCheckpoint(pl.Callback):
    def __init__(self, savepoints, dirpath, filename="{step}.ckpt"):
        self.savepoints = set(savepoints)
        self.dirpath = dirpath
        self.filename = filename
        self.lowest_val_loss = float("inf")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step in self.savepoints:
            filepath = os.path.join(self.dirpath, self.filename.format(step=step))
            trainer.save_checkpoint(filepath)
            print(f"Checkpoint saved at step {step}: {filepath}")

    def on_validation_epoch_end(self, trainer, pl_module):
        filepath = os.path.join(self.dirpath, "last.ckpt")
        trainer.save_checkpoint(filepath)

        if trainer.sanity_checking:
            return
        current_val_loss = pl_module.running_metrics["loss_total"]

        if current_val_loss is not None and current_val_loss < self.lowest_val_loss:
            self.lowest_val_loss = current_val_loss.clone().detach()
            filepath = os.path.join(self.dirpath, "lowest.ckpt")
            trainer.save_checkpoint(filepath)
            print(f"New minimum val loss detected: {current_val_loss}. Checkpoint saved as: {filepath}")


class ACCustomModelCheckpoint(pl.Callback):
    def __init__(self, savepoints, dirpath, filename="{step}.ckpt"):
        self.savepoints = set(savepoints)
        self.dirpath = dirpath
        self.filename = filename
        self.lowest_val_loss = float("inf")

    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     step = trainer.global_step
    #     if step in self.savepoints:
    #         filepath = os.path.join(self.dirpath, self.filename.format(step=step))
    #         trainer.save_checkpoint(filepath)
    #         print(f"Checkpoint saved at step {step}: {filepath}")

    def on_validation_epoch_end(self, trainer, pl_module):
        filepath = os.path.join(self.dirpath, "last.ckpt")
        trainer.save_checkpoint(filepath)


class EpochProgressLogger(pl.Callback):
    """Rank-0 only: log training epoch start/end to a local text file.

    Only writes concise epoch progress lines; does not log batch metrics or other info.
    """

    def __init__(self, dirpath: str, filename: str = "epoch_progress.log"):
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.filepath = self.dirpath / self.filename

    def _write_line(self, line: str):
        try:
            self.dirpath.mkdir(parents=True, exist_ok=True)
            with self.filepath.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            # Also print to console to ensure training progress is visible
            print(line)
        except Exception:
            # Silently ignore logging errors to avoid disrupting training
            pass

    def on_train_epoch_start(self, trainer, pl_module):
        # Only rank 0 writes progress
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return
        max_epochs = getattr(trainer, "max_epochs", None)
        # current_epoch is zero-based in Lightning
        current_epoch = int(getattr(trainer, "current_epoch", 0)) + 1
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(max_epochs, int) and max_epochs > 0:
            self._write_line(f"[{ts}] Epoch {current_epoch}/{max_epochs} START")
        else:
            self._write_line(f"[{ts}] Epoch {current_epoch} START")

    def on_train_epoch_end(self, trainer, pl_module):
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return
        max_epochs = getattr(trainer, "max_epochs", None)
        # current_epoch is still the same epoch index during end hook; display as completed
        current_epoch = int(getattr(trainer, "current_epoch", 0)) + 1
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(max_epochs, int) and max_epochs > 0:
            self._write_line(f"[{ts}] Epoch {current_epoch}/{max_epochs} END")
        else:
            self._write_line(f"[{ts}] Epoch {current_epoch} END")
