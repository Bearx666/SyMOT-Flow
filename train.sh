CUDA_VISIBLE_DEVICES=7 python3 mmd_mri.py \
--n_epochs 500 --batch_size_tr 30 --lr_t 1e-3 --lr_b 1e-3 --weight_decay 1e-5 \
--weight_ot 1e-3 --use_lr_schedular


CUDA_VISIBLE_DEVICES=7 python mmd_ct_mri.py \
--n_epochs 1000 --batch_size_tr 16 --n_batch 20 \
--lr_t 1e-3 --lr_b 1e-3 --weight_decay 0.0 --weight_ot 1.0 --use_lr_schedular \