for s in 5 6 7 8 9
do
    for task in expdecay poisson doublesingularity
    do
        for act in rational 
        do
            for h in 64
            do
                for n in 9 10 #15 14 13 12
                do
                    for k in 1 3 5 7 
                    do
                        for m in 31 15 7 5 3 1 0
                        do
                            python dd_gmg_1d.py --device 1 --task $task --act $act --seed $s --ep_adam 1000 --k $k --m $m --h $h --n $n --bsz 8
                        done 
                    done 
                done 
            done
        done
    done
done