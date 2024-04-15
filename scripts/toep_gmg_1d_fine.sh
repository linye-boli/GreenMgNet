for s in 0
do
    for task in cosine logarithm
    do
        for act in rational relu
        do
            for h in 50
            do
                for n in 13
                do
                    for k in 9 7
                    do
                        for m in 31 15 7 5 3 1 0
                        do
                            python toep_gmg_1d.py --device 3 --task $task --act $act --seed $s --ep_adam 2500 --k $k --m $m --h $h --n $n --bsz 20 --sch
                        done 
                    done 
                done 
            done
        done
    done
done