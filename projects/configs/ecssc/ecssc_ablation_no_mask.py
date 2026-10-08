# Ablation: no auxiliary moving-object supervision for E_obj.
_base_ = ['./ecssc.py']
work_dir = 'work_dirs/ecssc_ablation_no_mask'
model = dict(lambda_obj=0.0)
