


#python train_teacher.py --model resnet32x4 --ood tin --num_total_class 150 --num_ood_class 120
#python train_student.py --path_t ./save/models/resnet32x4_cifar100_lb:1.0_split:12345_uc:30_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s resnet8x4 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 120 --num_total_class 150 --include-labeled

#python train_teacher.py --model resnet32x4 --ood tin --num_total_class 100 --num_ood_class 10
#python train_student.py --path_t ./save/models/resnet32x4_cifar100_lb:1.0_split:12345_uc:90_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s resnet8x4 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 10 --num_total_class 100 --include-labeled

#python train_teacher.py --model resnet32x4 --ood tin --num_total_class 150 --num_ood_class 110
python train_student.py --path_t ./save/models/resnet32x4_cifar100_lb:1.0_split:12345_uc:20_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s resnet8x4 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 20 --num_total_class 100 --include-labeled

#python train_teacher.py --model resnet32x4 --ood tin --num_total_class 150 --num_ood_class 90
python train_student.py --path_t ./save/models/resnet32x4_cifar100_lb:1.0_split:12345_uc:30_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s resnet8x4 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 30 --num_total_class 100 --include-labeled

#python train_teacher.py --model resnet32x4 --ood tin --num_total_class 150 --num_ood_class 70
python train_student.py --path_t ./save/models/resnet32x4_cifar100_lb:1.0_split:12345_uc:40_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s resnet8x4 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 40 --num_total_class 100 --include-labeled


python train_student.py --path_t ./save/models/resnet32x4_cifar100_lb:1.0_split:12345_uc:50_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s resnet8x4 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 50 --num_total_class 100 --include-labeled

python train_student.py --path_t ./save/models/resnet32x4_cifar100_lb:1.0_split:12345_uc:60_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s resnet8x4 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 60 --num_total_class 100 --include-labeled

python train_student.py --path_t ./save/models/resnet32x4_cifar100_lb:1.0_split:12345_uc:70_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s resnet8x4 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 70 --num_total_class 100 --include-labeled

python train_student.py --path_t ./save/models/resnet32x4_cifar100_lb:1.0_split:12345_uc:80_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s resnet8x4 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 80 --num_total_class 100 --include-labeled


# WRN
python train_teacher.py --model wrn_40_2 --ood tin --num_total_class 100 --num_ood_class 20
python train_student.py --path_t ./save/models/wrn_40_2_cifar100_lb:1.0_split:12345_uc:20_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s wrn_40_1 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 50 --num_total_class 100 --include-labeled

python train_teacher.py --model wrn_40_2 --ood tin --num_total_class 100 --num_ood_class 40
python train_student.py --path_t ./save/models/wrn_40_2_cifar100_lb:1.0_split:12345_uc:40_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s wrn_40_1 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 70 --num_total_class 100 --include-labeled

python train_teacher.py --model wrn_40_2 --ood tin --num_total_class 100 --num_ood_class 60
python train_student.py --path_t ./save/models/wrn_40_2_cifar100_lb:1.0_split:12345_uc:60_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s wrn_40_1 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 80 --num_total_class 100 --include-labeled

python train_teacher.py --model wrn_40_2 --ood tin --num_total_class 100 --num_ood_class 80
python train_student.py --path_t ./save/models/wrn_40_2_cifar100_lb:1.0_split:12345_uc:80_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s wrn_40_1 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 80 --num_total_class 100 --include-labeled
