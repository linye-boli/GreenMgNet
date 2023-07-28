for ds in invdist darcy
do
    for i in 4 5
    do
        python fno/fourier_2d.py --dataset_nm $ds --trasub 3 --testsub 3 --clevel 0 --mlevel '-1' --seed $i --batch_size 16 --ntest 100 --device 0
        python fno/fourier_2d.py --dataset_nm $ds --trasub 3 --testsub 3 --clevel 0 --mlevel 0 --seed $i --batch_size 16 --ntest 100 --device 0
        python fno/fourier_2d.py --dataset_nm $ds --trasub 3 --testsub 3 --clevel 0 --mlevel 1 --seed $i --batch_size 16 --ntest 100 --device 0
        python fno/fourier_2d.py --dataset_nm $ds --trasub 3 --testsub 3 --clevel 0 --mlevel 2 --seed $i --batch_size 16 --ntest 100 --device 0
        python fno/fourier_2d.py --dataset_nm $ds --trasub 3 --testsub 3 --clevel 0 --mlevel 3 --seed $i --batch_size 16 --ntest 100 --device 0
    done 
done

