res=2049
ep=2500
device=1

for s in 1 3
do
    for task in logarithm cosine
    do
        # baseline fno
        python fourier_1d.py --device $device --task $task --seed $s --lr_adam 1e-3 --ep_adam 500 --sch --res $res

        for act in relu
        do
            for h in 128
            do  
                # toep_gl
                python green_mgnet.py --device $device --task $task --act $act --seed $s --mode toep_gl --lr_adam 1e-3 --ep_adam $ep --sch --h $h --res $res
                
                for k in 6 5 4 3 2 1
                do
                    for m in 0 1 3 7 15 31 63 127
                    do
                        # toep green mg
                        python green_mgnet.py --device $device --task $task --act $act --k $k --m $m --seed $s --mode toep_mg --lr_adam 1e-3 --ep_adam $ep --sch --h $h --res $res
                    done
                done
            done
        done
    done 
done