if [ $1 == "train" ] 
then
   echo "Training"
   python train.py $2 $3

elif [ $1 == "test" ] 
then
    echo "Testing"
    python test.py $2 $3
   
else
 echo "Argument not found"
 
fi