
for i in aw00 aw04 aw08 aw40 aw44 aw48 aw-40 aw-44 aw-48
do
    for j in 0 1 2 3 4 5 6 7 8
    do
    python R04sum_outputs.py  --alter=$i --case=$j
    done
done

