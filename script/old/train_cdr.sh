python3 train/train_motiondeblur.py --arch Uformer_B --batch_size 6 --gpu '4' \
    --train_ps 256 --train_dir /root/dataset/ReflectionRemoval/CDR_Split/train \
    --val_ps 256 --val_dir /root/dataset/ReflectionRemoval/CDR_Split/val --env _0706 \
    --mode deblur --nepoch 3000 --checkpoint 500 --dataset CDR --warmup