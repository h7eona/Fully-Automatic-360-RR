python3 train/train_motiondeblur.py --arch Uformer_B_noshift_nomodulator --batch_size 4 --gpu '8' \
    --train_ps 256 --train_dir /root/dataset/ReflectionRemoval/UnrealRR/CS_221105_split/train \
    --val_ps 256 --val_dir /root/dataset/ReflectionRemoval/UnrealRR/CS_221105_split/val --env _0706 \
    --mode deblur --nepoch 3000 --checkpoint 500 --dataset CS_221105_v2 --warmup

python3 train/train_motiondeblur.py --arch Uformer_B_noshift_nomodulator --batch_size 4 --gpu '8' \
    --train_ps 256 --train_dir /root/dataset/ReflectionRemoval/Syn80000/train \
    --val_ps 256 --val_dir /root/dataset/ReflectionRemoval/Syn80000/val --env _0706 \
    --mode deblur --nepoch 3000 --checkpoint 500 --dataset Syn80000_v2 --warmup