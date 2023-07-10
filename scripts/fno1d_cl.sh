for ds in lnabs cosine burgers poisson
do
    for i in 0 1 2 3 4 5 6 7 8 9
    do
        for c in 1 2 3 4 5
        do 
        python fno/fourier_1d.py --dataset_nm $ds --trasub 1 --testsub 1 --clevel $c --mlevel '-1' --seed $i --batch_size 20 --ntest 100 --device 1
        python fno/fourier_1d.py --dataset_nm $ds --trasub 1 --testsub 1 --clevel $c --mlevel 0 --seed $i --batch_size 20 --ntest 100 --device 1
        python fno/fourier_1d.py --dataset_nm $ds --trasub 1 --testsub 1 --clevel $c --mlevel 1 --seed $i --batch_size 20 --ntest 100 --device 1
        python fno/fourier_1d.py --dataset_nm $ds --trasub 1 --testsub 1 --clevel $c --mlevel 3 --seed $i --batch_size 20 --ntest 100 --device 1
        done 
    done 
done