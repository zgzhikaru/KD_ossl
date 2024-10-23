ROOT_DIR="results/"
tc=$1 #"wrn_40_2" #"resnet32x4"
arch=$2   #"wrn_40_1"   #"resnet8x4"

id_data="cifar100"
n_cls=100
ood="tin"
n_ood_cls=100
samples=500

lb=1.0
split=12345
trial=1 #0
tc_trial=0
method="kd"


KD_args="--distill "$method" -r 0.9 -a 0.1 -b 0"
#KD_args="--distill $method -a 0.0 -b 1.0 --hint_layer 4"

max_id_cls=100
max_ood_cls=200
max_samples=500

for min_ood_cls in `seq 100 -50 0`; #(100 50 0)
do
    samples_total=$((max_samples * (max_id_cls + min_ood_cls))) # (500,400,333)
    for n_ood_cls in `seq $min_ood_cls 50 200`;   #(50,100,150,200);
    do
        samples=$((samples_total/(max_id_cls + n_ood_cls)))
        ID_args="--dataset "$id_data" --num_classes "$max_id_cls
        OOD_args="--ood "$ood" --num_ood_class "$n_ood_cls

        st_save_name=M:$method"_T:"$tc"_arch:$arch""_ID:"$id_data"_ic:"$max_id_cls"_OOD:"$ood"_oc:"$n_ood_cls"_smp:"$samples"_lb:"$lb"_split:"$split"_trial:"$trial
        tc_save_name="M:supCE_arch:$tc""_ID:"$id_data"_ic:"$max_id_cls"_trial:"$tc_trial
        tc_path=$ROOT_DIR/teacher/model/$tc_save_name/$tc"_last.pth"

        python train_student.py --tc_path $tc_path --arch $arch $ID_args $OOD_args $KD_args --samples_per_cls $samples --trial $trial 
        python evaluate.py --model_path $ROOT_DIR/student/model/$st_save_name/$arch"_last.pth" --num_classes $max_id_cls --dataset $id_data --out_dir $ROOT_DIR/student/log/$st_save_name
    done
done
