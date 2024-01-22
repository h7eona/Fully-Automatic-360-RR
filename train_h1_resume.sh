python3 train/train.py \
    --img_height 256 \
    --img_width 512 \
    --train_reference_type full \
    --val_reference_type full \
    --train_dir /home/viplab/dataset/WACV28K_v3/train \
    --val_dir /home/viplab/dataset/WACV28K_v3/val \
    --save_dir /home/viplab/jhpark/2024_WACV/result \
    --real_dir /home/viplab/dataset/ECCV22_Real_NR_1K \
    --nepoch 30 \
    --checkpoint 500 \
    --warmup \
    --train_workers 8 \
    --batch_size 4 \
    --gpu '0,1' \
    --arch Uformer_B_laplacian \
    --dataset WACV28K_v3 \
    --env _h1_resume \
    --resume \
    --pretrain_weights /home/viplab/jhpark/2024_WACV/result/motiondeblur/WACV28K_v3/Uformer_B_laplacian_h1/models/model_epoch_20_6999.pth
    
    