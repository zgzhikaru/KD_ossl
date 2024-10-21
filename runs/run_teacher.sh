

ROOT_DIR="results/teacher/"
method="supCE"
tc="wrn_40_1" #"resnet8x4" #"resnet32x4"
id_data="cifar100"
n_cls_max=80    #100
trial=0

for n_cls in `seq $n_cls_max -20 20`;
do
    save_name="M:"$method"_arch:$tc""_ID:"$id_data"_ic:"$n_cls"_trial:"$trial
    
    python train_teacher.py --arch $tc --dataset $id_data --num_classes $n_cls
    for eval_n_cls in `seq $n_cls -20 20`;
    do
        python evaluate.py --model_path $ROOT_DIR"/model/"$save_name"/"$tc"_last.pth" --num_classes $eval_n_cls --dataset $id_data --out_dir $ROOT_DIR"/log/"$save_name
    done
done

