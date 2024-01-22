python3 train/train.py \
    --img_height 256 \
    --img_width 512 \
    --train_reference_type full \
    --val_reference_type full \
    --train_dir /home/viplab/dataset/syn_v2_wounreal/train \
    --val_dir /home/viplab/dataset/syn_v2_wounreal/val \
    --save_dir /home/viplab/jhpark/results/ \
    --nepoch 40 \
    --checkpoint 500 \
    --warmup \
    --batch_size 4 \
    --gpu '2,3' \
    --arch Uformer_B_noshift_laplacian \
    --dataset syn_v2_wounreal \
    --env _h1 \
    # --use_checkpoint \
    