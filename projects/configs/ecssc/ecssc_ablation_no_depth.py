# Ablation: no auxiliary depth supervision for E_ego.
_base_ = ['./ecssc.py']
work_dir = 'work_dirs/ecssc_ablation_no_depth'
model = dict(lambda_ego=0.0)
