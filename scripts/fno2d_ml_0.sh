for i in 0 1
do
    for ds in invdist darcy
    do
        python fno/fourier_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel 0 --mlevel '-1' --seed $i --batch_size 16 --ntest 100 --device 0
        python fno/fourier_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel 0 --mlevel 0 --seed $i --batch_size 16 --ntest 100 --device 0
        python fno/fourier_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel 0 --mlevel 1 --seed $i --batch_size 16 --ntest 100 --device 0
        python fno/fourier_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel 0 --mlevel 2 --seed $i --batch_size 16 --ntest 100 --device 0
        python fno/fourier_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel 0 --mlevel 3 --seed $i --batch_size 16 --ntest 100 --device 0
    done 
done

