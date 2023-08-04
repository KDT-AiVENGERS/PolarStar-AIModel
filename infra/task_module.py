# Modules About Torch, Numpy
import torch

# Modules About Pytorch Lightning
import lightning.pytorch as pl
# from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, STEP_OUTPUT

# Others
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class HFBertTask(pl.LightningModule):
    def __init__(self, tokenizer, predict_model=None, train_model=None, predict_target_cols=[], train_target_cols=[]) -> None:
        super().__init__()
        self.predict_target_cols = predict_target_cols
        self.train_target_cols = train_target_cols

        self.tokenizer = tokenizer

        self.predict_model = predict_model
        self.train_model = train_model
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.acc_func = None

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        if not (self.train_model or self.train_target_cols):
            print('No train_model or train_target_cols available!')
            return

        outputs = self.train_model(**batch)

        metrics = {
            'train_loss': outputs.loss
        }
        self.training_step_outputs.append(metrics)
        self.log_dict(metrics, prog_bar=True)

        return outputs.loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        if not (self.train_model or self.train_target_cols):
            print('No train_model or train_target_cols available!')
            return

        outputs = self.train_model(**batch)

        metrics = {
            'val_loss': outputs.loss
        }
        self.validation_step_outputs.append(metrics)
        self.log_dict(metrics, prog_bar=True)

        return outputs.loss

    def on_validation_epoch_end(self):
        if not (self.training_step_outputs and self.validation_step_outputs):
            return

        train_avg_loss = torch.stack([x["train_loss"]
            for x in self.training_step_outputs]).mean()
        metrics = {
            "train_avg_loss": train_avg_loss
        }
        self.log_dict(metrics)

        val_avg_loss = torch.stack([x["val_loss"]
            for x in self.validation_step_outputs]).mean()
        metrics = {
            "val_avg_loss": val_avg_loss
        }
        self.log_dict(metrics)

        print("\n" +
              (f'Epoch {self.current_epoch}, Avg. Training Loss: {train_avg_loss:.3f} ' +
               f'Avg. Validation Loss: {val_avg_loss:.3f}'), flush=True)

        self.training_step_outputs.clear()
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx) -> None:
        if not (self.train_model or self.train_target_cols):
            print('No train_model or train_target_cols available!')
            return

        outputs = self.train_model(**batch)

        metrics = {
            'test_loss': outputs.loss
        }
        self.log_dict(metrics, prog_bar=True)

        sentence_index, mask_token_index = (batch['labels'] != -100).nonzero(as_tuple=True)

        predicted_token_id = []
        for index, sentence in enumerate(sentence_index):
            if sentence >= len(predicted_token_id):
                predicted_token_id.append([])

            predicted_token_id[-1].append(self.tokenizer.decode(outputs.logits[sentence, mask_token_index[index]].argmax(axis=-1)))

        random_numbers = torch.randint(low=0, high=len(predicted_token_id), size=(int(len(batch['input_ids']) / 2),))

        original_token_id = self.tokenizer.batch_decode(batch['input_ids'])

        print('')
        for i in random_numbers:
            print(predicted_token_id[i], original_token_id[i])
            answers = batch['labels'][i]
            print(self.tokenizer.convert_ids_to_tokens(answers[answers != -100]))

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        if not (self.predict_model or self.predict_target_cols):
            print('No predict_model or predict_target_cols available!')

        outputs = self.predict_model(**batch)
        # pooler_outputs = outputs['pooler_output'] # these are the sentence embedding vectors (768 dim each)
        pooler_outputs = outputs[0][:, 0] # for torchscript
        outputs_concated = []
        for i in range(int(len(pooler_outputs) / len(self.predict_target_cols))):
            outputs_concated.append(torch.concat(list(pooler_outputs[i * len(self.predict_target_cols):(i + 1) * len(self.predict_target_cols)])))
            # Concatenating sentence embedding vectors from a job description

        return torch.stack(outputs_concated)

    def configure_optimizers(self):
        pass