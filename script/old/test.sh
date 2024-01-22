# python3 test/test_pano_vis.py --arch Uformer_B_noshift_nomodulator --gpus 8 --input_dir /root/dataset/ReflectionRemoval/Syn80000/val --result_dir ./results/vis0/ --weights /root/workplace/results/uniclus/motiondeblur/Syn80000/Uformer_B_noshift_nomodulator_AttnVis/models/model_best_e10.pth 
python3 test/test_pano_vis.py --arch Uformer_B_noshift_nomodulator --gpus 8 --input_dir /root/dataset/ReflectionRemoval/Syn80000/val --result_dir ./results/vis_e20/ --weights /root/workplace/results/uniclus/motiondeblur/Syn80000/Uformer_B_noshift_nomodulator_AttnVis/models/model_best_e20.pth
# python3 test/test_pano_vis.py --arch Uformer_B_noshift_nomodulator --gpus 8 --input_dir /root/dataset/ReflectionRemoval/ECCV22_Real_NR --result_dir ./results/vis_real/ --weights /root/workplace/results/uniclus/motiondeblur/Syn80000/Uformer_B_noshift_nomodulator_AttnVis/models/model_best_e10.pth 

### test on SIDD ###
# python3 test/test_sidd.py --input_dir ../datasets/denoising/sidd_val/ --result_dir ./results/denoising/SIDD/ --weights ./logs/denoising/SIDD/Uformer_B/models/model_best.pth 

### test on DND ###
# python3 test/test_dnd.py --input_dir ../datasets/denoising/dnd/input/ --result_dir ./results/denoising/DND/ --weights ./logs/denoising/SIDD/Uformer_B/models/model_best.pth 


### test on GoPro ###
# python3 test/test_gopro_hide.py --input_dir ../datasets/deblurring/GoPro/test/ --result_dir ./results/deblurring/GoPro/Uformer_B/ --weights ./logs/motiondeblur/GoPro/Uformer_B/models/model_best.pth

### test on HIDE ###
# python3 test/test_gopro_hide.py --input_dir ../datasets/deblurring/HIDE/test/ --result_dir ./results/deblurring/HIDE/Uformer_B/ --weights ./logs/motiondeblur/GoPro/Uformer_B/models/model_best.pth

### test on RealBlur ###
# python3 test/test_realblur.py --input_dir ../datasets/deblurring/ --result_dir ./results/deblurring/ --weights ./logs/motiondeblur/GoPro/Uformer_B/models/model_best.pth
