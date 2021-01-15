python train_on.py -e ConveyorEnv121 -s 20210113_0500 -n pipe10 -c pipe10
python train_on.py -e ConveyorEnv121 -s 20210113_0530 -n pipe15 -c pipe15
python train_on.py -e ConveyorEnv121 -s 20210113_0600 -n pipe20 -c pipe20
python train_on.py -e ConveyorEnv121 -s 20210113_0630 -n pipe25 -c pipe25
python train_on.py -e ConveyorEnv121 -s 20210113_0700 -n pipe30 -c pipe30
python train_on.py -e ConveyorEnv121 -s 20210113_0730 -n pipe35 -c pipe35
python train_on.py -e ConveyorEnv121 -s 20210113_0800 -n pipe40 -c pipe40
python train_on.py -e ConveyorEnv121 -s 20210113_0830 -n pipe45 -c pipe45
python train_on.py -e ConveyorEnv121 -s 20210113_0900 -n pipe50 -c pipe50
python plotcombiner_p.py -t terminate
python resultmaker_p.py -t pipe -n 100