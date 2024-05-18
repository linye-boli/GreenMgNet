for task in darcyrect poissonrect
do
    for s in 0 1 2 3 4
    do
        for act in rational
        do
            for h in 50
            do
                for n in 6
                do
                    for p in 0.2 0.001 0.005 0.03 0.05 0.1 0.15
                    do
                        for aug in aug2
                        do
                            python green_2d.py --device 1 --task $task --act $act --seed $s --ep_adam 1000 --h $h --n $n --bsz 20 --p $p --sch --aug $aug
                        done
                    done 
                done
            done
        done
    done
done