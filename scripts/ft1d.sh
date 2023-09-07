for s in 16 8 4 2 1
do
    for ds in lnabs cosine burgers poisson
    do
        for c in 0 1 2 3 4
        do 
            for m in '-1' 0 1 2 3 4
            do
                python ft/ft_1d.py --dataset_nm $ds --trasub $s --testsub $s --clevel $c --mlevel $m --seed $2 --batch_size 16 --ntest 200 --device $1
            done
        done 
    done
done 
