for i in 1991 2001 2011 2021 2031
do
    for s in 1 2 4 8 16
    do
        for ds in lnabs, cosine, burgers
        do
            for c in 0 1 2 3
            do
            python fno/fourier_1d.py --dataset_nm $ds --trasub $s --testsub $s --clevel $c --mlevel 0 --res_type null --bw 0 --seed $i --batch_size 20 --ntest 100
            python fno/fourier_1d.py --dataset_nm $ds --trasub $s --testsub $s --clevel $c --mlevel 0 --res_type diag --bw 0 --seed $i --batch_size 20 --ntest 100
            python fno/fourier_1d.py --dataset_nm $ds --trasub $s --testsub $s --clevel $c --mlevel 0 --res_type band --bw 3 --seed $i --batch_size 20 --ntest 100
            done
        done
    done
done 
