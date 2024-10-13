

ROOT_DIR="results/teacher/"
method="supCE"
tc="resnet32x4"
id_data="cifar100"
n_cls=100
trial=0

for n_cls in (100,80,60,40,20);
do
    save_name=$ROOT_DIR"M:"$method"_arch:$tc""_ID:"$id_data"_ic:"$n_cls"_trial:"$trial
    
    python train_teacher.py --model resnet32x4 --dataset cifar100 --num_classes n_cls
    python evaluate.py --path $save_name
end


arch="resnet8x4"
dataset="cifar100"
ood="tin"
n_ood_cls=100
samples=500

lb=1.0
split=12345
trial=0
method="kd"


KD_args="--distill "$method" -r 0.9 -a 0.1 -b 0"


n_total=200
max_n_ood=180
min_n_cls=$((n_total - max_n_ood))
for n_ood_cls in (100,120,140,160,180);
do
    n_cls=$((n_total - n_ood_cls))
    ID_args="--dataset "$dataset" --num_classes "$n_cls
    OOD_args="--ood "$ood" --num_ood_class "$n_ood_cls

    tc_name=$ROOT_DIR"M:"$method"_arch:$tc""_ID:"$id_data"_ic:"$n_cls"_trial:"$trial
    save_name=$ROOT_DIR"M:"$method"_T:"$tc"_arch:$arch""_ID:"$id_data"_ic:"$n_cls"_OOD:"$ood"_oc:"$n_ood_cls"_smp:"$samples"_lb:"$lb"_split:"$split"_trial:"$trial
    
    python train_student.py --tc_path $ROOT_DIR --arch $arch $ID_args $OOD_args $KD_args #--samples_per_cls $samples 
    python evaluate.py --path $save_name --num_classes $min_n_cls
end

max_n_id_cls=100
max_samples=500
samples_total=$((max_samples * max_n_id_cls))
for n_ood_cls in (100,125,150,175,200);
do
    samples=$((samples_total/n_ood_cls))
    ID_args="--dataset "$dataset" --num_classes "$max_n_id_cls
    OOD_args="--ood "$ood" --num_ood_class "$n_ood_cls

    python train_student.py --tc_path $ROOT_DIR --arch $arch $ID_args $OOD_args $KD_args --samples_per_cls $samples 
    python evaluate.py --path $save_name --num_classes $max_n_id_cls
end