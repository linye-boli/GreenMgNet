for s in 0
do
    for act in relu #rational 
    do
        for task in poisson expdecay
        do
            for h in 50
            do
                for n in 9
                do
                    for k in 7 5 3 1
                    do
                        for m in 0 1 3 5 7 15 31
                        do
                            python dd_gmg_1d.py --device 1 --task $task --act $act --seed $s --ep_adam 2500 --k $k --m $m --h $h --n $n --bsz 20 --sch
                        done 
                    done 
                done 
            done
        done
    done
done