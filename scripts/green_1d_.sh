for s in 0 1 2
do
    for act in rational
    do
        for task in logarithm #cosine 
        do
            for h in 50
            do
                for n in 9
                do
                    for p in 1.0 #0.01 0.03 0.05 0.07 0.1 0.15 0.25 0.30 0.4 0.5 0.7 0.9
                    do
                        for aug in none aug2 
                        do 
                            python green_1d.py --device 1 --task $task --act $act --seed $s --ep_adam 10000 --h $h --p $p --n $n --bsz 200 --sch --aug $aug
                        done
                    done
                done
            done
        done
    done
done