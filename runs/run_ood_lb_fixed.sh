ROOT_DIR="results/"
tc=$1 #"wrn_40_2" #"resnet32x4"
arch=$2   #"wrn_40_1"   #"resnet8x4"

id_data="cifar100"
ood="tin"

split=12345
trial=0
method="kd" #"hint"

KD_args="--distill "$method" -r 0.9 -a 0.1 -b 0"
#KD_args="--distill $method -a 0.0 -b 1.0 --hint_layer 4"


samples_total=75000
full_samples=500 #num_labels
n_id_cls=100
max_ood_cls=200
min_ood_cls=50


for samples_total in 75000 60000; #100000 50000; 
do
num_labels=$((samples_total * n_id_cls/(n_id_cls + max_ood_cls)))
echo "Number of labels:" $num_labels
#tc_samples=$num_labels
min_ood_cls=$((samples_total/full_samples - n_id_cls))

for n_ood_cls in `seq $max_ood_cls -50 $min_ood_cls`; 
do
    ID_args="--dataset "$id_data" --num_classes "$n_id_cls
    OOD_args="--ood "$ood" --num_ood_class "$n_ood_cls
    SPLIT_args="--num_samples "$samples_total" --num_labels "$num_labels

    st_save_name=M:$method"_T:"$tc"_arch:$arch""_ID:"$id_data"_ic:"$n_id_cls"_OOD:"$ood"_oc:"$n_ood_cls"_lb:"$num_labels"_total:"$samples_total"_split:"$split"_trial:"$trial
    tc_save_name="M:supCE_arch:$tc""_ID:"$id_data"_ic:"$n_id_cls"_total:"$num_labels"_trial:"$trial
    #tc_save_name="M:supCE_arch:$tc""_ID:"$id_data"_ic:"$n_id_cls"_total:"$full_samples"_trial:"$trial
    tc_path=$ROOT_DIR/teacher/model/$tc_save_name/$tc"_last.pth"
        
    python train_student.py --tc_path $tc_path --arch $arch $ID_args $OOD_args $SPLIT_args $KD_args
    python evaluate.py --model_path $ROOT_DIR/student/model/$st_save_name/$arch"_last.pth" --num_classes $n_id_cls --dataset $id_data --out_dir $ROOT_DIR/student/log/$st_save_name
done
done