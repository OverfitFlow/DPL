import os
import copy
from torchmetrics.functional import f1_score
import pytorch_lightning as pl

from clip.config import ex
from clip.modules import CLIPransformerSS
from clip.datamodules.multitask_datamodule import MTDataModule
import shutil

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)

    model = CLIPransformerSS(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    #logger = pl.loggers.TensorBoardLogger(
    #    _config["log_dir"],
    #    name=f'{exp_name}_seed{_config["seed"]}}',
    #)
    logger = pl.loggers.CSVLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}',
    )
    target_dir = './result_model_files' + '/' + _config["log_dir"] +'/' + f'{exp_name}_seed{_config["seed"]}' + '/'
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        os.makedirs(target_dir)
    else:
        os.makedirs(target_dir)
    shutil.copy2('./clip/modules/clip_missing_aware_prompt_module.py', target_dir)
    shutil.copy2('./clip/modules/vision_transformer_prompts.py', target_dir)
    shutil.copy2('./clip/config.py', target_dir)

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]
#     from pytorch_lightning.profiler import SimpleProfiler
#     profiler = SimpleProfiler()
    
    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )
    print(_config["batch_size"], _config["per_gpu_batchsize"], num_gpus, _config["num_nodes"])
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
#         profiler=profiler,
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
