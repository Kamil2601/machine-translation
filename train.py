import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(
        self,
        model,
        train_dataLoader,
        loss_fn,
        optimizer,
        val_dataLoader=None,
        padding_index=0,
        sos_index=1,
        teacher_forcing_ratio=0.5,
        device=device,
    ) -> None:
        self.model = model.to(device)
        self.train_dataLoader = train_dataLoader
        self.val_dataLoader = val_dataLoader
        self.loss_fn = loss_fn
        self.padding_index = padding_index
        self.sos_index = sos_index
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = device
        self.optimizer = optimizer

    def loss_value(self, output, target):
        C = output.shape[-1]

        output_flat = output.view(-1, C)
        target_flat = target.view(-1)

        loss = self.loss_fn(output_flat, target_flat)

        return loss

    def train_batch(self, encoder_input, decoder_input, target):
        teacher_forcing = np.random.random() < self.teacher_forcing_ratio

        output = self.model(
            encoder_input,
            dec_input_batch=decoder_input,
            teacher_forcing=teacher_forcing,
            sos_index=self.sos_index,
            out_length=target.shape[1],
        )

        loss = self.loss_value(output, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0

        epoch_loss = 0
        batch_losses = []

        i = 1

        for enc_input, dec_input, target in self.train_dataLoader:
            i += 1
            enc_input = enc_input.to(self.device)
            dec_input = dec_input.to(self.device)
            target = target.to(self.device)

            batch_loss = self.train_batch(
                encoder_input=enc_input, decoder_input=dec_input, target=target
            )

            epoch_loss += batch_loss * len(enc_input)
            batch_losses.append(batch_loss)

        size = len(self.train_dataLoader.dataset)

        return epoch_loss / size, batch_losses

    def validation(self):
        size = len(self.val_dataLoader.dataset)
        self.model.eval()

        test_loss = 0

        with torch.inference_mode():
            for enc_input, _, target in self.val_dataLoader:
                enc_input = enc_input.to(self.device)
                target = target.to(self.device)

                out = self.model(
                    enc_input,
                    teacher_forcing=False,
                    sos_index=self.sos_index,
                    out_length=target.shape[1],
                )
                test_loss += self.loss_value(out, target).item() * len(enc_input)

            return test_loss / size

    def train(self, n_epochs, verbose = True):
        for epoch in range(n_epochs):
            epoch_loss, batch_losses = self.train_one_epoch()

            if self.val_dataLoader != None:
                val_loss = self.validation()

                if verbose:
                    print(f"Epoch {epoch + 1 :< 4}  training loss: {epoch_loss:>8f} | validation loss: {val_loss:>8f}")\
                    
            else:
                if verbose:
                    print(f"Epoch {epoch + 1 :< 10}  training loss: {epoch_loss:>8f}")