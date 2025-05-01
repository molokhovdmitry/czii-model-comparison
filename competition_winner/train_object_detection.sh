# for FOLD in 0
# do
#   python train_od.py \
#   --adam_beta1=0.95 --adam_beta2=0.99 \
#   --average_tokens_across_devices=True \
#   --dataloader_num_workers=4 \
#   --dataloader_persistent_workers=True \
#   --dataloader_pin_memory=True \
#   --ddp_find_unused_parameters=True \
#   --early_stopping=2 \
#   --fold=$FOLD \
#   --learning_rate=5e-4 \
#   --max_grad_norm=3 \
#   --ddp_find_unused_parameters=True \
#   --model_name=segresnetv2 \
#   --num_train_epochs=1 \
#   --per_device_train_batch_size=1 \
#   --pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
#   --scale_limit=0.05 \
#   --use_6_classes=True \
#   --use_instance_crops=True \
#   --use_random_crops=True \
#   --use_stride4=False \
#   --warmup_steps=0 \
#   --weight_decay=0.01 \
#   --y_rotation_limit=5 \
#   --x_rotation_limit=5 \
#   --seed=$FOLD \
#   --interpolation_mode=1 \
#   --tf32=False \
#   --validate_on_rot90=False \
#   --num_crops_per_study=1 \
#   --bf16=False \
#   --fp16=True \
#   --max_steps=12 \
#   --save_steps=2 \
#   --split_strategy=split_data_into_folds \
#   --valid_spatial_num_tiles=3 \
#   --valid_depth_num_tiles=1
# done

for FOLD in 0
do
  python train_od.py \
  --average_tokens_across_devices=True \
  --dataloader_num_workers=4 \
  --dataloader_persistent_workers=True \
  --dataloader_pin_memory=True \
  --tf32=False \
  --ddp_find_unused_parameters=True \
  --early_stopping=2 \
  --fold=$FOLD \
  --learning_rate=3e-3 \
  --max_grad_norm=3 \
  --ddp_find_unused_parameters=True \
  --model_name=dynunet  \
  --num_train_epochs=1 \
  --per_device_train_batch_size=1 \
  --scale_limit=0.05 \
  --use_6_classes=True \
  --use_instance_crops=True \
  --use_random_crops=True \
  --use_stride4=False \
  --warmup_steps=0 \
  --weight_decay=0.0001 \
  --x_rotation_limit=5 \
  --x_rotation_limit=5 \
  --seed=$FOLD \
  --interpolation_mode=1 \
  --tf32=False \
  --validate_on_rot90=False \
  --num_crops_per_study=1 \
  --bf16=False \
  --fp16=True \
  --max_steps=12 \
  --save_steps=2 \
  --split_strategy=split_data_into_folds \
  --valid_spatial_num_tiles=3 \
  --valid_depth_num_tiles=1
done