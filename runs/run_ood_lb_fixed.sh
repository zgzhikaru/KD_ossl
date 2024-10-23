ROOT_DIR="results/"
tc=$1 #"wrn_40_2" #"resnet32x4"
arch=$2   #"wrn_40_1"   #"resnet8x4"

id_data="cifar100"
ood="tin"

#lb=1.0
split=12345
trial=0
method="kd"

KD_args="--distill "$method" -r 0.9 -a 0.1 -b 0"


samples_total=75000
n_id_cls=100
max_ood_cls=200
min_ood_cls=50
max_samples=500
num_labels=$((samples_total * n_id_cls/(n_id_cls + max_ood_cls)))

echo "Number of labels:" $num_labels

for samples_total in 75000, 60000; 
do
for n_ood_cls in `seq $min_ood_cls 50 $max_ood_cls`;   #(100,125,150,175,200);
do
    #samples=$((samples_total/n_ood_cls))
    ID_args="--dataset "$id_data" --num_classes "$n_id_cls
    OOD_args="--ood "$ood" --num_ood_class "$n_ood_cls
    SPLIT_args="--num_samples "$samples_total" --num_labels "$num_labels

    python train_student.py --tc_path $ROOT_DIR --arch $arch $ID_args $OOD_args $SPLIT_args $KD_args
    python evaluate.py --path $save_name --num_classes $n_id_cls
done
done