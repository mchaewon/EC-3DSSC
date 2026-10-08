# Ablation: E_ego branch only (E_obj features zeroed, moving-object loss off).
_base_ = ['./ecssc.py']
work_dir = 'work_dirs/ecssc_ablation_ego_only'
model = dict(ablation='ego_only')
