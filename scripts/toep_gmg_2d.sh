for task in invdist
do
    for act in rational
    do
        for h in 64
        do
            for n in 5
            do
                for k in 5 3 1 0
                do
                    for m in 7 5 3 1 0
                    do
                        python toep_gmg_2d.py --device 1 --task $task --act $act --seed 0 --ep_adam 1000 --k $k --m $m --h $h --n $n --bsz 4
                    done 
                done 
            done 
        done
    done
done