## Download and Formatting Instructions for EndoSlam
1) wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/cd2rtzm23r-1.zip
2) unzip "*.zip" all zipped files inside folders
3) Restructure the folders so that the Cameras folder contains UnityCam only. **Remove OlympusCam and PillCam, they cannot be used as there is either no pose information or no ground truth.**

## Instructions for Running EndoSlam Preprocessing
1) Either use parameters in endoslam_preprocess.py or use prepare_endoslam.sh. Will need larger GPUs if using the bash script. **If you need ground truth depth map labels, only preprocess UnityCam dataset**
3) Once all data has been preprocessed, run train_test_split.py with required arguments.
4) endoslam_dataset.py contains custom Dataset class to read in the train_test splits for Monst3R
5) visualize.py allows you to visualize the preprocessed camera frames and associated depth maps.
