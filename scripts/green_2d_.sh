for s in 0 1 2 3 4
do
    for task in poissonrect
    do
        for act in rational
        do
            for h in 50
            do
                for n in 6
                do
                    for p in 0.001 0.005 0.03 0.05 0.1 0.15 0.2
                    do
                        for aug in aug2 none
                        do
                            python green_2d.py --device 0 --task $task --act $act --seed $s --ep_adam 1000 --h $h --n $n --bsz 20 --p $p --sch --aug $aug
                        done
                    done 
                done
            done
        done
    done
done