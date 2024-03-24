for s in 5 7 9 #0 2 4
do
    for task in logarithm poisson cosine
    do
        for act in rational 
        do
            for h in 64
            do
                for n in 9 #15 14 13 12
                do
                    for k in 1 3 5 7 
                    do
                        for m in 31 15 7 5 3 1 0
                        do
                            python dd_gmg_1d.py --device 0 --task $task --act $act --seed $s --ep_adam 1000 --k $k --m $m --h $h --n $n --bsz 8
                        done 
                    done 
                done 
            done
        done
    done
done