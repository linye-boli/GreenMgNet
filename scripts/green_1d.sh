for s in 0 1 2 3 4
do
    for task in logarithm poisson cosine
    do
        for act in rational
        do
            for h in 64
            do
                for n in 9 10 11
                do
                    python green_1d.py --device 1 --task $task --act $act --seed $s --ep_adam 1000 --h $h --n $n --bsz 8
                done
            done
        done
    done
done