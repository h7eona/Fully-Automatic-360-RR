python3 train/train.py \
    --img_height 256 \
    --img_width 512 \
    --train_reference_type full \
    --val_reference_type full \
    --train_dir /home/viplab/dataset/Syn50000/train \
    --val_dir /home/viplab/dataset/Syn50000/val \
    --save_dir /home/viplab/jhpark/results/ \
    --nepoch 40 \
    --checkpoint 500 \
    --warmup \
    --batch_size 4 \
    --gpu '2,3' \
    --arch Uformer_B_noshift_nomodulator_laplacian \
    --dataset Syn50000 \
    --env _h1 \
    --resume \
    --pretrain_weights /home/viplab/jhpark/results/motiondeblur/Syn50000/Uformer_B_noshift_nomodulator_laplacian_h1/models/model_latest.pth
    