for s in 0 1 2 3 4
do
    for task in poisson invdist
    do
        for act in relu rational
        do
            for h in 64
            do
                for n in 6
                do
                    python green_2d.py --device 0 --task $task --act $act --seed $s --ep_adam 1000 --h $h --n $n --bsz 4
                done
            done
        done
    done
done