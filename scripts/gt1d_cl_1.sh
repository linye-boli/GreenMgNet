for ds in lnabs cosine burgers poisson
do
    for i in 2 3
    do
        for c in 1 2 3 4
        do 
        python gt/gt_1d.py --dataset_nm $ds --trasub 16 --testsub 16 --clevel $c --mlevel '-1' --seed $i --batch_size 16 --ntest 200 --device 1
        python gt/gt_1d.py --dataset_nm $ds --trasub 16 --testsub 16 --clevel $c --mlevel 0 --seed $i --batch_size 16 --ntest 200 --device 1
        python gt/gt_1d.py --dataset_nm $ds --trasub 16 --testsub 16 --clevel $c --mlevel 1 --seed $i --batch_size 16 --ntest 200 --device 1
        python gt/gt_1d.py --dataset_nm $ds --trasub 16 --testsub 16 --clevel $c --mlevel 2 --seed $i --batch_size 16 --ntest 200 --device 1
        python gt/gt_1d.py --dataset_nm $ds --trasub 16 --testsub 16 --clevel $c --mlevel 3 --seed $i --batch_size 16 --ntest 200 --device 1
        python gt/gt_1d.py --dataset_nm $ds --trasub 16 --testsub 16 --clevel $c --mlevel 4 --seed $i --batch_size 16 --ntest 200 --device 1
        done 
    done 
done