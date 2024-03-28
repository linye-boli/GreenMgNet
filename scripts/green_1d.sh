for s in 0 1 2 3 4
do
    for act in relu
    do
        for task in expdecay poisson doubleholes cosine logarithm
        do
            for h in 50
            do
                for n in 11
                do
                    python green_1d.py --device 3 --task $task --act $act --seed $s --ep_adam 2500 --h $h --n $n --bsz 20 --sch
                done
            done
        done
    done
done