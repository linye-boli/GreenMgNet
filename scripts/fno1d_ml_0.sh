for ds in lnabs cosine poisson burgers
do
    for i in 0 1
    do
        python fno/fourier_1d.py --dataset_nm $ds --trasub 16 --testsub 8 --clevel 0 --mlevel '-1' --seed $i --batch_size 16 --ntest 200
        python fno/fourier_1d.py --dataset_nm $ds --trasub 16 --testsub 8 --clevel 0 --mlevel 0 --seed $i --batch_size 16 --ntest 200
        python fno/fourier_1d.py --dataset_nm $ds --trasub 16 --testsub 8 --clevel 0 --mlevel 1 --seed $i --batch_size 16 --ntest 200
        python fno/fourier_1d.py --dataset_nm $ds --trasub 16 --testsub 8 --clevel 0 --mlevel 2 --seed $i --batch_size 16 --ntest 200
        python fno/fourier_1d.py --dataset_nm $ds --trasub 16 --testsub 8 --clevel 0 --mlevel 3 --seed $i --batch_size 16 --ntest 200
        python fno/fourier_1d.py --dataset_nm $ds --trasub 16 --testsub 8 --clevel 0 --mlevel 3 --seed $i --batch_size 16 --ntest 200
    done 
done

