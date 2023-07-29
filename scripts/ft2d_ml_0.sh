for ds in darcy invdist
do
    for i in 0 1
    do
        python ft/ft_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel 0 --mlevel '-1' --seed $i --batch_size 2 --ntest 100
        python ft/ft_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel 0 --mlevel 0 --seed $i --batch_size 2 --ntest 100
        python ft/ft_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel 0 --mlevel 1 --seed $i --batch_size 2 --ntest 100
        python ft/ft_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel 0 --mlevel 2 --seed $i --batch_size 2 --ntest 100
        python ft/ft_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel 0 --mlevel 3 --seed $i --batch_size 2 --ntest 100
    done 
done

