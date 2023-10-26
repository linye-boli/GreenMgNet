for s in 5 3 2
do
    for ds in darcy invdist
    do
        case $s in 
            2) 
            for c in 0 1 2 3
            do
                for m in '-1' 0 1 2 3
                do
                    python ft/ft_2d.py --dataset_nm $ds --trasub $s --testsub $s --clevel $c --mlevel $m --seed $2 --batch_size 2 --ntest 100 --epochs 200 --lr 0.00025 --device $1
                done 
            done
            ;;
            3 | 5)
            for c in 0 1 2 3
            do
                for m in '-1' 0 1 2 3
                do
                    python ft/ft_2d.py --dataset_nm $ds --trasub $s --testsub $s --clevel $c --mlevel $m --seed $2 --batch_size 2 --ntest 100 --epochs 200 --lr 0.00025 --device $1
                done 
            done
            esac
    done 
done 
