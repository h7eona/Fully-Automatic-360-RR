python3 train/train_motiondeblur.py --arch Uformer_S_noshift_nomodulator --batch_size 1 --gpu '7' \
    --train_ps 256 --train_dir /root/dataset/ReflectionRemoval/Syn80000/train \
    --val_ps 256 --val_dir /root/dataset/ReflectionRemoval/Syn80000/val --env _0706 \
    --mode deblur --nepoch 3000 --checkpoint 500 --dataset Syn80000_S_hor --warmup