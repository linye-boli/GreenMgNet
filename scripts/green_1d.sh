for s in 5 6 7 8 9 #0 1 2 3 4
do
    for task in expdecay poisson doublesingularity
    do
        for act in rational
        do
            for h in 64
            do
                for n in 9 10
                do
                    python green_1d.py --device 0 --task $task --act $act --seed $s --ep_adam 1000 --h $h --n $n --bsz 8
                done
            done
        done
    done
done