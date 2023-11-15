for ds in ns_V1e-3
    do
    for s in 1
    do
        for c in 0 1
        do 
            for m in 0 1 2 3 
            do
                python fno/fourier_2dt.py --dataset_nm $ds --trasub $s --testsub $s --clevel $c --mlevel $m --seed $2 --batch_size 10 --ntrain 4000 --ntest 800 --device $1
            done         
        done
    done 
done 

for ds in ns_V1e-4
    do
    for s in 1
    do
        for c in 0 1
        do 
            for m in 0 1 2 3 
            do
                python fno/fourier_2dt.py --dataset_nm $ds --trasub $s --testsub $s --clevel $c --mlevel $m --seed $2 --batch_size 10 --ntrain 8000 --ntest 1600 --device $1
            done         
        done
    done 
done 
