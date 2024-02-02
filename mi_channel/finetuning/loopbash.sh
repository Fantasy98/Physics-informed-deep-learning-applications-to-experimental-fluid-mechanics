

for i in $(seq 1 8);
do 
    echo "We at $i"
    python -u a2_weight_tune.py | tee "$i""log.out"
    
    sleep 30
done 
