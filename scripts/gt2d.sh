for ds in darcy invdist
do
    for s in 1 2 3 4 5
    do
        case $s in 
            1)
            for c in 3 4
            do
                for m in '-1' 0 1 2 3 4
                do
                    python gt/gt_2d.py --dataset_nm $ds --trasub $s --testsub $s --clevel $c --mlevel $m --seed $2 --batch_size 2 --ntest 100 --epochs 200 --lr 0.00025 --device $1
                done 
            done 
            ;;
            2) 
            for c in 1 2 3 4
            do
                for m in '-1' 0 1 2 3 4
                do
                    python gt/gt_2d.py --dataset_nm $ds --trasub $s --testsub $s --clevel $c --mlevel $m --seed $2 --batch_size 2 --ntest 100 --epochs 200 --lr 0.00025 --device $1
                done 
            done
            ;;
            3 | 4 | 5)
            for c in 0 1 2 3 4
            do
                for m in '-1' 0 1 2 3 4
                do
                    python gt/gt_2d.py --dataset_nm $ds --trasub $s --testsub $s --clevel $c --mlevel $m --seed $2 --batch_size 2 --ntest 100 --epochs 200 --lr 0.00025 --device $1
                done 
            done
            esac
    done 
done 
