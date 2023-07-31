for i in 6 7
do
    for ds in darcy invdist
    do
        python ft/ft_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel 0 --mlevel '-1' --seed $i --batch_size 2 --ntest 100 --epochs 200 --lr 0.00025
        python ft/ft_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel 0 --mlevel 0 --seed $i --batch_size 2 --ntest 100 --epochs 200 --lr 0.00025
        python ft/ft_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel 0 --mlevel 1 --seed $i --batch_size 2 --ntest 100 --epochs 200 --lr 0.00025
        python ft/ft_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel 0 --mlevel 2 --seed $i --batch_size 2 --ntest 100 --epochs 200 --lr 0.00025
        python ft/ft_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel 0 --mlevel 3 --seed $i --batch_size 2 --ntest 100 --epochs 200 --lr 0.00025
    done 
done

