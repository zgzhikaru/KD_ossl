

ROOT_DIR="results/teacher/"
method="supCE"
tc=$1  #wrn_40_2
id_data="cifar100"
n_cls_max=100
trial=0
max_samples=500

for samples in 20000; #25000
do
for tc in 'wrn_40_1'; do
    save_name="M:"$method"_arch:$tc""_ID:"$id_data"_ic:"$n_cls_max"_total:"$samples"_trial:"$trial
    
    python train_teacher.py --arch $tc --dataset $id_data --num_classes $n_cls_max --num_samples $samples
    for eval_n_cls in `seq $n_cls -20 20`;
    do
        python evaluate.py --model_path $ROOT_DIR"/model/"$save_name"/"$tc"_last.pth" --num_classes $eval_n_cls --dataset $id_data --out_dir $ROOT_DIR"/log/"$save_name
    done
done
done

