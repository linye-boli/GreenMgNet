for s in 0
do
    for act in relu rational
    do
        for task in expdecay poisson doublesingularity cosine logarithm
        do
            for h in 64
            do
                for n in 12
                do
                    python green_1d.py --device 3 --task $task --act $act --seed $s --ep_adam 1000 --h $h --n $n --bsz 8
                done
            done
        done
    done
done