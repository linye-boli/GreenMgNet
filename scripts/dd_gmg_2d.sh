for s in 1 2 3 4 0
do
    for task in poisson
    do
        for act in relu rational 
        do
            for h in 64
            do
                for n in 6
                do
                    for k in 3 2 1
                    do
                        for m in 7 3 2 1
                        do
                            python dd_gmg_2d.py --device 1 --task $task --act $act --seed $s --ep_adam 1000 --k $k --m $m --h $h --n $n --bsz 4
                        done 
                    done 
                done 
            done
        done
    done
done