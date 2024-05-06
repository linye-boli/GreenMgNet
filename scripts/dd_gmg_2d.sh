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
                    for k in 3 2 1
                    do
                        for m in 7 5 3 1 0
                        do
                            for aug in aug2
                            do
                                python dd_gmg_2d.py --device 1 --task $task --act $act --seed $s --ep_adam 1000 --k $k --m $m --h $h --n $n --bsz 20 --sch --aug $aug
                            done                            
                        done 
                    done 
                done 
            done
        done
    done
done