for i in 6 7
do
    for ds in darcy invdist
    do
        for c in 1 2 3
        do 
        python fno/fourier_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel $c --mlevel '-1' --seed $i --batch_size 16 --ntest 100 --device 3
        python fno/fourier_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel $c --mlevel 0 --seed $i --batch_size 16 --ntest 100 --device 3
        python fno/fourier_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel $c --mlevel 1 --seed $i --batch_size 16 --ntest 100 --device 3
        python fno/fourier_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel $c --mlevel 2 --seed $i --batch_size 16 --ntest 100 --device 3
        done 
    done 
done