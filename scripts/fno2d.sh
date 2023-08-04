for s in 3 4 5
do
    for ds in darcy invdist
    do
        for c in 0 1 2 3 4
        do 
            for m in '-1' 0 1 2 3 4
            do
                python fno/fourier_2d.py --dataset_nm $ds --trasub $s --testsub $s --clevel $c --mlevel $m --seed $2 --batch_size 16 --ntest 100 --device $1
            done 
        done
    done 
done 
