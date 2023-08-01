# Modules About Torch
import torch

# Modules About Pytorch Lightning
import lightning.pytorch as pl

from serving.data_module import HFBertDataModule
from serving.task_module import HFBertTask

# Inference
def inference(Tokenizer, Model, model_name, target_columns = ['자격요건', '우대조건', '복지', '회사소개', '주요업무']):
  predict_target_cols = target_columns

  tokenizer = Tokenizer.from_pretrained(model_name)
  data_module = HFBertDataModule(
      tokenizer=tokenizer,
      max_batch_size=15,
      data_path='data/JobDescription/pre_result_2.csv',
      predict_target_cols=predict_target_cols,
  )

  model = Model.from_pretrained(model_name)
  task = HFBertTask(tokenizer=tokenizer, predict_model=model, predict_target_cols=predict_target_cols)

  trainer = pl.Trainer(
    logger=False
  )

  predicted_embedding_vectors = trainer.predict(task, datamodule=data_module) # this list contains tensors of each output of batch running
  concatenated_embedding_vectors = torch.concat(predicted_embedding_vectors, dim=-2)

  return concatenated_embedding_vectors.shape

def inference_torchscript(Tokenizer, model_jit, model_name, target_columns = ['자격요건', '우대조건', '복지', '회사소개', '주요업무']):
  predict_target_cols = target_columns

  tokenizer = Tokenizer.from_pretrained(model_name)
  data_module = HFBertDataModule(
    tokenizer=tokenizer,
    max_batch_size=15,
    data_path='data/JobDescription/pre_result_2.csv',
    predict_target_cols=predict_target_cols,
  )

  model = torch.jit.load(model_jit)
  task = HFBertTask(tokenizer=tokenizer, predict_model=model, predict_target_cols=predict_target_cols)

  trainer = pl.Trainer(
    logger=False
  )

  predicted_embedding_vectors = trainer.predict(task, datamodule=data_module)
  concatenated_embedding_vectors = torch.concat(predicted_embedding_vectors, dim=-2)

  return concatenated_embedding_vectors.shape


