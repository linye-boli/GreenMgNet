for ds in lnabs cosine burgers poisson
do
    for i in 0 1 2 3 4 5 6 7 8 9
    do
        python fno/fourier_1d.py --dataset_nm $ds --trasub 1 --testsub 1 --clevel 0 --mlevel '-1' --seed $i --batch_size 20 --ntest 100
        python fno/fourier_1d.py --dataset_nm $ds --trasub 1 --testsub 1 --clevel 0 --mlevel 0 --seed $i --batch_size 20 --ntest 100
        python fno/fourier_1d.py --dataset_nm $ds --trasub 1 --testsub 1 --clevel 0 --mlevel 1 --seed $i --batch_size 20 --ntest 100
        python fno/fourier_1d.py --dataset_nm $ds --trasub 1 --testsub 1 --clevel 0 --mlevel 2 --seed $i --batch_size 20 --ntest 100
        python fno/fourier_1d.py --dataset_nm $ds --trasub 1 --testsub 1 --clevel 0 --mlevel 3 --seed $i --batch_size 20 --ntest 100
        python fno/fourier_1d.py --dataset_nm $ds --trasub 1 --testsub 1 --clevel 0 --mlevel 4 --seed $i --batch_size 20 --ntest 100
        python fno/fourier_1d.py --dataset_nm $ds --trasub 1 --testsub 1 --clevel 0 --mlevel 5 --seed $i --batch_size 20 --ntest 100
    done 
done

