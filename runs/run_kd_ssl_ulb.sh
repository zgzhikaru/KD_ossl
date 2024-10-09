# srd
#python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill srd -a 0 -b 1 --ood tin

# kd
python train_teacher.py --model resnet32x4 --lb_prop 0.5 --split_seed 12345 --ood tin

python train_student.py --path_t ./save/models/resnet32x4_cifar100_lb:0.5_split:12345_uc:0_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s resnet8x4 --distill kd -r 0.1 -a 0.9 -b 0 --ood tin --lb_prop 0.5 --split_seed 12345 --include-labeled --ood tin --num_ood_class 200
