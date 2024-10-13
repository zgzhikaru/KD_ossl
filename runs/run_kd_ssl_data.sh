
# KD 
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --distill srd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 0 --include-labeled

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill srd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 0 --include-labeled

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --distill srd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 150 --include-labeled

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill srd -r 0.9 -a 0.1 -b 0 --ood tin --num_ood_class 150 --include-labeled


## SRD resnet
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --distill srd -r 1 -a 1 -b 1 --ood tin --num_ood_class 200 --include-labeled

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --distill srd -r 1 -a 1 -b 1 --ood tin --num_ood_class 100 --include-labeled

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --distill srd -r 1 -a 1 -b 1 --ood tin --num_ood_class 50 --include-labeled

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --distill srd -r 1 -a 1 -b 1 --ood tin --num_ood_class 0 --include-labeled

# SRD wrn
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill srd -r 1 -a 1 -b 1 --ood tin --num_ood_class 200 --include-labeled

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill srd -r 1 -a 1 -b 1 --ood tin --num_ood_class 100 --include-labeled

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill srd -r 1 -a 1 -b 1 --ood tin --num_ood_class 50 --include-labeled

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill srd -r 1 -a 1 -b 1 --ood tin --num_ood_class 0 --include-labeled


## SRD
#python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --distill srd -a 1 -b 1 --ood tin --num_ood_class 200 --include-labeled

#python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --distill srd -a 1 -b 1 --ood tin --num_ood_class 100 --include-labeled

#python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --distill srd -a 1 -b 1 --ood tin --num_ood_class 50 --include-labeled

#python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill srd -a 1 -b 1 --ood tin --num_ood_class 200 --include-labeled

#python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill srd -a 1 -b 1 --ood tin --num_ood_class 100 --include-labeled

#python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill srd -a 1 -b 1 --ood tin --num_ood_class 50 --include-labeled
