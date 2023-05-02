echo "Arg 1: $1";
echo "Arg 2: $2";
echo "Arg 3: $3";

if [ $1 == "train" ]
then
    echo "train"
    python train_save.py --mode $1 --train_file $2 --val_file $3
elif [ $1 == "test" ]
then
    echo "test"
    python train_save.py --mode $1 --test_file $2 --out_file $3
else
    echo "arguments not recognized"
fi