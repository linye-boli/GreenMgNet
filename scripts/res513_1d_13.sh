for s in 1 3
do
    for task in logarithm laplace schrodinger cosine
    do
        # baseline fno
        python fourier_1d.py --device 0 --task $task --seed $s --lr_adam 1e-3 --ep_adam 1000 --sch --res 513

        for act in rational relu
        do
            for h in 64 128
            do
                
                # baseline gl 
                python green_mgnet.py --device 0 --task $task --act $act --seed $s --mode gl --lr_adam 1e-3 --ep_adam 1000 --sch --h $h --res 513
                # toep_gl
                python green_mgnet.py --device 0 --task $task --act $act --seed $s --mode toep_gl --lr_adam 1e-3 --ep_adam 1000 --sch --h $h --res 513
                # lr_gl
                for r in 8 16 32
                do
                    python green_mgnet.py --device 1 --task $task --act $act --r $r --seed $s --mode lr_gl --lr_adam 1e-3 --ep_adam 1000 --sch --h $h --res 513
                done

                for k in 3 1
                do
                    for m in 0 3 7 15 31 63
                    do
                        # toep green mg
                        python green_mgnet.py --device 0 --task $task --act $act --k $k --m $m --seed $s --mode toep_mg --lr_adam 1e-3 --ep_adam 1000 --sch --h $h --res 513
                    done
                done

                for k in 3 1
                do
                    for m in 0 3 7 15 31 63
                    do
                        # dd green mg
                        python green_mgnet.py --device 0 --task $task --act $act --k $k --m $m --seed $s --mode dd_mg --lr_adam 1e-3 --ep_adam 1000 --sch --h $h --res 513
                    done
                done
            done
        done
    done 
done