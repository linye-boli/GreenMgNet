for s in 0
do
    for act in relu
    do
        for task in poisson expdecay
        do
            for h in 50
            do
                for n in 13
                do
                    for k in 9 7
                    do
                        for m in 0 1 3 7
                        do
                            python dd_gmg_1d.py --device 2 --task $task --act $act --seed $s --ep_adam 2500 --k $k --m $m --h $h --n $n --bsz 20 --sch
                        done 
                    done 
                done 
            done
        done
    done
done