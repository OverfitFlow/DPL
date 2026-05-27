device=0
missing_pattern=both
missing_ratio=0.9
exp_name=hate/train/DPL_both09
exp_name_test=hate/test/DPL_both09
contrast_coef=7
arc_m=0.15
arc_m_tm=0.25
arc_m_im=0.25
data_path=datasets/Hatefull_Memes
task=task_finetune_hatememes

CUDA_VISIBLE_DEVICES=${device} \
python run.py with data_root=${data_path} \
    per_gpu_batchsize=4 num_gpus=1 num_nodes=1 ${task} \
    missing_ratio="{'train': ${missing_ratio}, 'val': ${missing_ratio}, 'test': ${missing_ratio}}" \
    missing_type="{'train': ${missing_pattern}, 'val': ${missing_pattern}, 'test': ${missing_pattern}}" \
    seed=0 \
    exp_name=${exp_name} \
    use_pl=42 \
    arc_s=32.0 arc_s_tm=32.0 arc_s_im=32.0 \
    arc_m=${arc_m} arc_m_tm=${arc_m_tm} arc_m_im=${arc_m_im} \
    contrast_coef=${contrast_coef} contrast_temp=1.0 contrast_temp_base=1.0 contrast_mode="all"


CUDA_VISIBLE_DEVICES=0 \
python run.py with data_root=${data_path} \
    per_gpu_batchsize=4 num_gpus=1 num_nodes=1 ${task} \
    missing_ratio="{'train': ${missing_ratio}, 'val': ${missing_ratio}, 'test': ${missing_ratio}}" \
    missing_type="{'train': ${missing_pattern}, 'val': ${missing_pattern}, 'test': ${missing_pattern}}" \
    seed=0 \
    exp_name=${exp_name_test} \
    arc_s=32.0 arc_s_tm=32.0 arc_s_im=32.0 \
    arc_m=${arc_m} arc_m_tm=${arc_m_tm} arc_m_im=${arc_m_im} \
    contrast_coef=${contrast_coef} contrast_temp=1.0 contrast_temp_base=1.0 contrast_mode="all" \
    test_only=True \
    load_path=$(find result/${exp_name}_seed0/version_0/checkpoints/ -type f -name "epoch*")
