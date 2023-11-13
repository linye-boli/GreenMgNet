for s in 1
do
    for c in 4
    do 
        for m in 0 1
        do
            for ds in NS_V1e-3 NS_V1e-4 NS_V1e-5
            do
                python gt/gt_2dt.py --dataset_nm $ds --trasub $s --testsub $s --clevel $c --mlevel $m --seed $2 --batch_size 2 --ntest 200 --device $1
            done         
        done
    done 
done 
