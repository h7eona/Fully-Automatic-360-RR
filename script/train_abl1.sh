python3 train/train.py \
    --img_height 256 \
    --img_width 512 \
    --train_reference_type full \
    --val_reference_type full \
    --train_dir /home/viplab/dataset/1_geo_absorption/train \
    --val_dir /home/viplab/dataset/1_geo_absorption/val \
    --save_dir /home/viplab/jhpark/res_abl/ \
    --real_dir /home/viplab/dataset/ECCV22_Real_Paper \
    --nepoch 40 \
    --checkpoint 500 \
    --warmup \
    --train_workers 8 \
    --batch_size 4 \
    --gpu '0,1' \
    --arch Uformer_B_laplacian \
    --dataset 1_geo_absorption \
    --env _h1
    

