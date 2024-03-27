for s in 0 1 2
do
    for task in poisson
    do
        for act in tanh rational 
        do
            for h in 64
            do
                for n in 6
                do
                    for k in 1 2 3
                    do
                        for m in 7 5 3 1 0
                        do
                            python dd_gmg_1d.py --device 1 --task $task --act $act --seed $s --ep_adam 1000 --k $k --m $m --h $h --n $n --bsz 4
                        done 
                    done 
                done 
            done
        done
    done
done