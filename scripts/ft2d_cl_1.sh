for i in 2 3
do
    for ds in darcy invdist
    do
        for c in 1 2 3
        do 
        python ft/ft_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel $c --mlevel '-1' --seed $i --batch_size 2 --ntest 100 --epochs 200 --lr 0.00025 --log_dir /home/omnisky/linye/research/MINO/logs  --cfg_path /home/omnisky/linye/research/MINO/cfgs --dataset_path /home/omnisky/linye/research/pde_data --device 0
        python ft/ft_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel $c --mlevel 0 --seed $i --batch_size 2 --ntest 100 --epochs 200 --lr 0.00025 --log_dir /home/omnisky/linye/research/MINO/logs  --cfg_path /home/omnisky/linye/research/MINO/cfgs --dataset_path /home/omnisky/linye/research/pde_data --device 0
        python ft/ft_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel $c --mlevel 1 --seed $i --batch_size 2 --ntest 100 --epochs 200 --lr 0.00025 --log_dir /home/omnisky/linye/research/MINO/logs  --cfg_path /home/omnisky/linye/research/MINO/cfgs --dataset_path /home/omnisky/linye/research/pde_data --device 0
        python ft/ft_2d.py --dataset_nm $ds --trasub 5 --testsub 5 --clevel $c --mlevel 2 --seed $i --batch_size 2 --ntest 100 --epochs 200 --lr 0.00025 --log_dir /home/omnisky/linye/research/MINO/logs  --cfg_path /home/omnisky/linye/research/MINO/cfgs --dataset_path /home/omnisky/linye/research/pde_data --device 0
        done 
    done 
done