for s in 3 4
do
    for act in rational 
    do
        for task in airy1d boundary1d # schrodinger1d poisson1d
        do
            for h in 50
            do
                for n in 9 #15 14 13 12
                do
                    for k in 1 2 3
                    do
                        for m in 0 1 3 7 15 31 # 65 31 7 
                        do
                            for aug in aug2 none
                            do
                            python dd_gmg_1d.py --device 2 --task $task --act $act --seed $s --ep_adam 10000 --k $k --m $m --h $h --n $n --bsz 200 --sch --aug $aug
                            done 
                        done 
                    done 
                done 
            done
        done
    done
done