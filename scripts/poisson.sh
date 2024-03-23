for h in 64 128
do
    for n in 15 14 13
    do
        for k in 1 3 5 7
        do
            for m in 31 15 7 3
            do
                python toep_gmg_1d.py --device 2 --task poisson --act rational --seed 0 --ep_adam 1000 --k $k --m $m --h $h --n $n
            done 
        done 
    done 
done