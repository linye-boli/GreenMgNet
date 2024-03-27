for s in 0 1 2 3 4
do
    for act in relu rational 
    do
        for task in expdecay poisson doubleholes
        do
            for h in 50
            do
                for n in 12 #15 14 13 12
                do
                    for k in 7 5 3 1
                    do
                        for m in 65 31 15 7 3 1 0
                        do
                            python dd_gmg_1d.py --device 2 --task $task --act $act --seed $s --ep_adam 2500 --k $k --m $m --h $h --n $n --bsz 20 --sch
                        done 
                    done 
                done 
            done
        done
    done
done