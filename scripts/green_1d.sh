for s in 0 1 2 3 4
do
    for task in expdecay poisson doublesingularity cosine logarithm
    do
        for act in relu
        do
            for h in 64
            do
                for n in 12
                do
                    python green_1d.py --device 1 --task $task --act $act --seed $s --ep_adam 1000 --h $h --n $n --bsz 8
                done
            done
        done
    done
done