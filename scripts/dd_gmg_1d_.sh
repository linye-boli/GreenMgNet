for s in 0 2 4
do
    for act in relu rational 
    do
        for task in expdecay poisson doubleholes
        do
            for h in 50
            do
                for n in 11 #15 14 13 12
                do
                    for k in 3 2 1
                    do
                        for m in 0 1 3 5 7 15 # 65 31 7 
                        do
                            python dd_gmg_1d.py --device 1 --task $task --act $act --seed $s --ep_adam 2500 --k $k --m $m --h $h --n $n --bsz 20 --sch
                        done 
                    done 
                done 
            done
        done
    done
done