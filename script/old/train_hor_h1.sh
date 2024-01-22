python3 train/train_motiondeblur.py --arch Uformer_B_noshift_nomodulator --batch_size 2 --gpu '3' \
    --train_ps 256 --train_dir /root/dataset/ReflectionRemoval/Syn80000/train \
    --val_ps 256 --val_dir /root/dataset/ReflectionRemoval/Syn80000/val --env _0706 \
    --mode deblur --nepoch 3000 --checkpoint 500 --dataset Syn80000_B_hor_h1 --warmup