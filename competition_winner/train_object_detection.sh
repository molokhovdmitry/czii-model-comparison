for FOLD in 0
do
  torchrun --standalone --nproc-per-node=1 train_od.py \
  --adam_beta1=0.95 --adam_beta2=0.99 \
  --average_tokens_across_devices=True \
  --dataloader_num_workers=4 \
  --dataloader_persistent_workers=True \
  --dataloader_pin_memory=True \
  --ddp_find_unused_parameters=True \
  --early_stopping=15 \
  --fold=$FOLD \
  --learning_rate=5e-5 \
  --max_grad_norm=3 \
  --ddp_find_unused_parameters=True \
  --model_name=segresnetv2 \
  --num_train_epochs=3 \
  --per_device_train_batch_size=2 \
  --pretrained_backbone_path=/home/ubuntu/Kaggle-2024-CryoET/pretrained/wholeBody_ct_segmentation/models/model.pt \
  --scale_limit=0.05 \
  --use_6_classes=True \
  --use_instance_crops=True \
  --use_random_crops=True \
  --use_stride4=False \
  --warmup_steps=16 \
  --weight_decay=0.01 \
  --y_rotation_limit=5 \
  --x_rotation_limit=5 \
  --seed=$FOLD \
  --interpolation_mode=1 \
  --validate_on_rot90=True
done

for FOLD in 0
do
  torchrun --standalone --nproc-per-node=1 train_od.py \
  --average_tokens_across_devices=True \
  --dataloader_num_workers=4 \
  --dataloader_persistent_workers=True \
  --dataloader_pin_memory=True \
  --ddp_find_unused_parameters=True \
  --early_stopping=25 \
  --fold=$FOLD \
  --learning_rate=3e-4 \
  --max_grad_norm=3 \
  --ddp_find_unused_parameters=True \
  --model_name=dynunet  \
  --num_train_epochs=3 \
  --per_device_train_batch_size=4 \
  --scale_limit=0.05 \
  --use_6_classes=True \
  --use_instance_crops=True \
  --use_random_crops=True \
  --use_stride4=False \
  --warmup_steps=16 \
  --weight_decay=0.0001 \
  --x_rotation_limit=5 \
  --x_rotation_limit=5 \
  --seed=$FOLD \
  --interpolation_mode=1 \
  --validate_on_rot90=True
done
