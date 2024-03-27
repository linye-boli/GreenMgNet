for s in 0 1 2 3 4
do
    for task in cosine logarithm
    do
        for act in relu
        do
            for h in 64
            do
                for n in 10
                do
                    for k in 7 5 3 1 0
                    do
                        for m in 31 15 7 5 3 1 0
                        do
                            python toep_gmg_1d.py --device 1 --task $task --act $act --seed $s --ep_adam 1000 --k $k --m $m --h $h --n $n --bsz 8
                        done 
                    done 
                done 
            done
        done
    done
done