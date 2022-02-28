for i in 2080
do
    for j in 0 1 2 3 4 5 6 7 8
    do
        #for h in aw00 aw02 aw04 aw-40 aw40 aw42 aw44 aw-42 aw-44
        for h in aw00
        do
        python sum_test_alter_aw_sc1.py --year=$i --case=$j --alter=$h
        done
    done
done


for i in 2080
do
    for j in 0 1 2 3 4 5 6 7 8
    do
        #for h in aw00 aw02 aw04 aw-40 aw40 aw42 aw44 aw-42 aw-44
        for h in aw00
        do
        python sum_test_alter_aw_sc2.py --year=$i --case=$j --alter=$h
        done
    done
done

