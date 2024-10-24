#!bash

bash runs/run_teacher_smp.sh
bash runs/run_ood_lb_fixed.sh resnet32x4 resnet8x4
bash runs/run_ood_lb_fixed.sh wrn_40_2 wrn_40_1