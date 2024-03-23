for s in 0 1 2 3 4
do
    for act in relu, rational
    do
        for h in 128
        do
            for k in 1 2 3 4 5 6 7
            do
                for m in 0 1 3 7 15 31 63 127
                do
                    # toep green mg
                    python dd_gmgn_1d.py --n 13 --task logarithm --device 1 --k $k --m $m
                done
            done
        done
    done
done