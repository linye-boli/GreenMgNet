for s in 0
do
    for task in poisson invdist
    do
        for act in rational
        do
            for h in 50
            do
                for n in 6
                do
                    for p in 0.005 0.03 0.05 0.1 0.15 0.25 0.30 0.4 0.5 1.0
                    do
                        python green_2d.py --device 0 --task $task --act $act --seed $s --ep_adam 2500 --h $h --n $n --bsz 20 --p $p --sch
                    done 
                done
            done
        done
    done
done