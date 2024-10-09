

#python train_teacher.py --model resnet32x4 --ood tin --num_total_class 200 --num_ood_class 100
#python train_student.py --path_t ./save/models/resnet32x4_cifar100_lb:1.0_split:12345_uc:0_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s resnet8x4 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 100 --num_total_class 200 --include-labeled

python train_teacher.py --model resnet32x4 --ood tin --num_total_class 200 --num_ood_class 150
python train_student.py --path_t ./save/models/resnet32x4_cifar100_lb:1.0_split:12345_uc:50_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s resnet8x4 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 150 --num_total_class 200 --include-labeled

python train_teacher.py --model resnet32x4 --ood tin --num_total_class 200 --num_ood_class 180
python train_student.py --path_t ./save/models/resnet32x4_cifar100_lb:1.0_split:12345_uc:80_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s resnet8x4 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 180 --num_total_class 200 --include-labeled

python train_teacher.py --model resnet32x4 --ood tin --num_total_class 200 --num_ood_class 120
python train_student.py --path_t ./save/models/resnet32x4_cifar100_lb:1.0_split:12345_uc:20_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s resnet8x4 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 120 --num_total_class 200 --include-labeled


#python train_teacher.py --model resnet32x4 --ood tin --num_total_class 150 --num_ood_class 50
python train_student.py --path_t ./save/models/resnet32x4_cifar100_lb:1.0_split:12345_uc:0_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s resnet8x4 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 50 --num_total_class 150 --include-labeled

#python train_teacher.py --model resnet32x4 --ood tin --num_total_class 150 --num_ood_class 100
python train_student.py --path_t ./save/models/resnet32x4_cifar100_lb:1.0_split:12345_uc:50_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s resnet8x4 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 100 --num_total_class 150 --include-labeled

python train_teacher.py --model resnet32x4 --ood tin --num_total_class 150 --num_ood_class 120
python train_student.py --path_t ./save/models/resnet32x4_cifar100_lb:1.0_split:12345_uc:30_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s resnet8x4 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 120 --num_total_class 150 --include-labeled

python train_teacher.py --model resnet32x4 --ood tin --num_total_class 150 --num_ood_class 80
python train_student.py --path_t ./save/models/resnet32x4_cifar100_lb:1.0_split:12345_uc:70_lr:0.05_decay:0.0005_trial:0/ckpt_epoch_240.pth --model_s resnet8x4 --distill kd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 80 --num_total_class 150 --include-labeled