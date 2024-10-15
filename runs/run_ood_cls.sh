ROOT_DIR="results/"
tc="resnet32x4"
arch="resnet8x4"

id_data="cifar100"
n_cls=100
ood="tin"
n_ood_cls=100
samples=500

lb=1.0
split=12345
trial=0
method="kd"


KD_args="--distill "$method" -r 0.9 -a 0.1 -b 0"

n_total=200
min_n_cls=20
for n_ood_cls in `seq 100 20 $((n_total - min_n_cls))`;    #(100,120,140,160,180);
do
    n_cls=$((n_total - n_ood_cls))
    ID_args="--dataset "$id_data" --num_classes "$n_cls
    OOD_args="--ood "$ood" --num_ood_class "$n_ood_cls
    tc_save_name="M:supCE_arch:$tc""_ID:"$id_data"_ic:"$n_cls"_trial:"$trial
    tc_path=$ROOT_DIR/teacher/model/$tc_save_name/$tc"_last.pth"
    
    st_save_name=M:$method"_T:"$tc"_arch:$arch""_ID:"$id_data"_ic:"$n_cls"_OOD:"$ood"_oc:"$n_ood_cls"_smp:"$samples"_lb:"$lb"_split:"$split"_trial:"$trial
    
    python train_student.py --tc_path $tc_path --arch $arch $ID_args $OOD_args $KD_args #--samples_per_cls $samples 
    for cls in `seq $n_cls -20 $min_n_cls`; do
        python evaluate.py --path $ROOT_DIR/student/model/$st_save_name/$arch"_last.pth" --num_classes $cls --dataset $id_data --out_dir $ROOT_DIR/student/log/$st_save_name
    done
done

: '
max_n_id_cls=100
max_samples=500
samples_total=$((max_samples * max_n_id_cls))
for n_ood_cls in `seq 100 25 200`;   #(100,125,150,175,200);
do
    samples=$((samples_total/n_ood_cls))
    ID_args="--dataset "$dataset" --num_classes "$max_n_id_cls
    OOD_args="--ood "$ood" --num_ood_class "$n_ood_cls

    python train_student.py --tc_path $ROOT_DIR --arch $arch $ID_args $OOD_args $KD_args --samples_per_cls $samples 
    python evaluate.py --path $save_name --num_classes $max_n_id_cls
done
'