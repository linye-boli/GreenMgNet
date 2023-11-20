for s in 5 3 2 1
do
    for c in 3 2 1 0
    do
        for m in '-1' 0 1 2 3
        do
            python ft/ft_2d_profile.py --dataset_nm 'invdist' --trasub $s --testsub $s --clevel $c --mlevel $m --seed $2 --batch_size 2 --ntest 100 --device $1 --epochs 5
        done
    done 
done 
