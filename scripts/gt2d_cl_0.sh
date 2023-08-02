for i in 0 1
do
    for ds in darcy invdist
    do
        for c in 1 2 3
        do 
        python gt/gt_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel $c --mlevel '-1' --seed $i --batch_size 8 --ntest 100 --epochs 200 --lr 0.00025 --device 0
        python gt/gt_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel $c --mlevel 0 --seed $i --batch_size 8 --ntest 100 --epochs 200 --lr 0.00025 --device 0
        python gt/gt_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel $c --mlevel 1 --seed $i --batch_size 8 --ntest 100 --epochs 200 --lr 0.00025 --device 0
        python gt/gt_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel $c --mlevel 2 --seed $i --batch_size 8 --ntest 100 --epochs 200 --lr 0.00025 --device 0
        python gt/gt_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel $c --mlevel 3 --seed $i --batch_size 8 --ntest 100 --epochs 200 --lr 0.00025 --device 0
        done 
    done 
done

