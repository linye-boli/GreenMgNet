res=513
ep=1500

for s in 1 3
do
    for task in schrodinger laplace 
    do
        # baseline fno
        python fourier_1d.py --device 1 --task $task --seed $s --lr_adam 1e-3 --ep_adam 500 --sch --res $res

        for act in relu
        do
            for h in 128
            do
                
                # baseline gl 
                python green_mgnet.py --device 1 --task $task --act $act --seed $s --mode gl --lr_adam 1e-3 --ep_adam $ep --sch --h $h --res $res
                
                # lr_gl
                # for r in 8 16 32
                # do
                #     python green_mgnet.py --device 1 --task $task --act $act --r $r --seed $s --mode lr_gl --lr_adam 1e-3 --ep_adam 1000 --sch --h $h --res 2049
                # done

                for k in 5 3 1
                do
                    for m in 0 1 3 7 15 31
                    do
                        # dd green mg
                        python green_mgnet.py --device 1 --task $task --act $act --k $k --m $m --seed $s --mode dd_mg --lr_adam 1e-3 --ep_adam $ep --sch --h $h --res $res
                    done
                done
            done
        done
    done 
done