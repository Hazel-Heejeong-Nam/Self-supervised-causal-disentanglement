num_workers=8
batch_size=512
seed=1
sup=weaksup

for c_lr in 0.0001 0.001; do 
    for c_beta in 4 1; do 
        python main.py --sup $sup --schedule True --seed $seed --c_beta $c_beta --c_lr $c_lr --num_workers $num_workers --batch_size $batch_size > labelLR_${l_lr}_causalityLR_${c_lr}.txt
    done
done



# sup=weaksup
# python main.py --sup $sup --schedule True --seed $seed --c_beta $c_beta --l_beta $l_beta --l_dag_w1 $l_dag_w1 --l_dag_w2 $l_dag_w2 --l_lr $l_lr --c_lr $c_lr --num_workers $num_workers --batch_size $batch_size > labelLR_${l_lr}_causalityLR_${c_lr}.txt




# l_dag_w1=3
# l_dag_w2=0.5
# python main.py --sup $sup --schedule True --seed $seed --c_beta $c_beta --l_beta $l_beta --l_dag_w1 $l_dag_w1 --l_dag_w2 $l_dag_w2 --l_lr $l_lr --c_lr $c_lr --num_workers $num_workers --batch_size $batch_size > labelLR_${l_lr}_causalityLR_${c_lr}.txt


# l_dag_w1=1.5
# l_dag_w2=0.25
# python main.py --sup $sup --schedule True --seed $seed --c_beta $c_beta --l_beta $l_beta --l_dag_w1 $l_dag_w1 --l_dag_w2 $l_dag_w2 --l_lr $l_lr --c_lr $c_lr --num_workers $num_workers --batch_size $batch_size > labelLR_${l_lr}_causalityLR_${c_lr}.txt


# l_dag_w1=0
# l_dag_w2=0
# python main.py --sup $sup --schedule True --seed $seed --c_beta $c_beta --l_beta $l_beta --l_dag_w1 $l_dag_w1 --l_dag_w2 $l_dag_w2 --l_lr $l_lr --c_lr $c_lr --num_workers $num_workers --batch_size $batch_size > labelLR_${l_lr}_causalityLR_${c_lr}.txt
