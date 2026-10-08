# Ablation: E_obj branch only (E_ego features zeroed, depth loss off).
_base_ = ['./ecssc.py']
work_dir = 'work_dirs/ecssc_ablation_obj_only'
model = dict(ablation='obj_only')
