for ds in cosine burgers poisson lnabs 
do
    for i in 0
    do
        python fno/fourier_1d.py --dataset_nm $ds --trasub 8 --testsub 8 --clevel 0 --mlevel '-1' --seed $i --batch_size 20 --ntest 100
        python fno/fourier_1d.py --dataset_nm $ds --trasub 8 --testsub 8 --clevel 0 --mlevel 0 --seed $i --batch_size 20 --ntest 100
        python fno/fourier_1d.py --dataset_nm $ds --trasub 8 --testsub 8 --clevel 0 --mlevel 1 --seed $i --batch_size 20 --ntest 100
        python fno/fourier_1d.py --dataset_nm $ds --trasub 8 --testsub 8 --clevel 0 --mlevel 2 --seed $i --batch_size 20 --ntest 100
        # python fno/fourier_1d.py --dataset_nm $ds --trasub 8 --testsub 8 --clevel 0 --mlevel 3 --seed $i --batch_size 20 --ntest 100
        # python fno/fourier_1d.py --dataset_nm $ds --trasub 8 --testsub 8 --clevel 0 --mlevel 4 --seed $i --batch_size 20 --ntest 100
        # python fno/fourier_1d.py --dataset_nm $ds --trasub 8 --testsub 8 --clevel 0 --mlevel 5 --seed $i --batch_size 20 --ntest 100
    done 
done

