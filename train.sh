c_lr=0.0001
num_workers=8
batch_size=512

for l_lr in 0.001 0.0003 0.0001 0.00003 0.00001 0.000003 0.000001; do
    python main.py --l_lr $l_lr --c_lr $c_lr --num_workers $num_workers --batch_size $batch_size > labelLR_${l_lr}_causalityLR_${c_lr}.txt
done

#0.00003 0.00001 0.000003 0.000001