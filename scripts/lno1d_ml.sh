for ds in lnabs cosine burgers poisson
do
    python lno/lno_1d.py --dataset_nm $ds --trasub 1 --testsub 1 --clevel 0 --mlevel '-1' --seed 0 --batch_size 5 --ntest 100
    python lno/lno_1d.py --dataset_nm $ds --trasub 1 --testsub 1 --clevel 0 --mlevel 0 --seed 0 --batch_size 5 --ntest 100
    python lno/lno_1d.py --dataset_nm $ds --trasub 1 --testsub 1 --clevel 0 --mlevel 1 --seed 0 --batch_size 5 --ntest 100
    python lno/lno_1d.py --dataset_nm $ds --trasub 1 --testsub 1 --clevel 0 --mlevel 2 --seed 0 --batch_size 5 --ntest 100
    python lno/lno_1d.py --dataset_nm $ds --trasub 1 --testsub 1 --clevel 0 --mlevel 3 --seed 0 --batch_size 5 --ntest 100
    python lno/lno_1d.py --dataset_nm $ds --trasub 1 --testsub 1 --clevel 0 --mlevel 4 --seed 0 --batch_size 5 --ntest 100
    python lno/lno_1d.py --dataset_nm $ds --trasub 1 --testsub 1 --clevel 0 --mlevel 5 --seed 0 --batch_size 5 --ntest 100
done