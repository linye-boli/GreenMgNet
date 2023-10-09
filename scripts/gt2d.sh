for s in 1 2 5 4 3
do
    for ds in darcy invdist
    do
        for c in 2 3 4
        do 
            for m in '-1' 0 1 2 3 4
            do
                python gt/gt_2d.py --dataset_nm $ds --trasub $s --testsub $s --clevel $c --mlevel $m --seed $2 --batch_size 2 --ntest 100 --epochs 200 --lr 0.00025 --device $1
            done 
        done
    done 
done 
