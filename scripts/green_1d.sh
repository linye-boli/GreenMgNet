for s in 0 1 2 3 4 5
do
    for act in rational relu
    do
        for task in poisson cosine logarithm
        do
            for h in 50
            do
                for n in 9
                do
                    for p in 1.0 0.01 0.03 0.05 0.07 0.1 0.15 0.25 0.30 0.4 0.5 0.7 0.9 
                    do
                        python green_1d.py --device 0 --task $task --act $act --seed $s --ep_adam 1000 --h $h --p $p --n $n --bsz 100 --sch
                    done
                done
            done
        done
    done
done