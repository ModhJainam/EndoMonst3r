## Download and Formatting Instructions for EndoSlam
wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/cd2rtzm23r-1.zip

unzip "*.zip" all zipped files inside folders

Restructure the folders so that the Cameras folder contains HighCam, LowCam, MiroCam, and UnityCam only -> Remove OlympusCam and PillCam, they cannot be used as there is either no pose information or no ground truth.

Edit Camera organ folders to match names in 3D Scanner folders (e.g. Change Colon-III(L-shaped) to Colon-III)
Edit HighCam and LowCam calibration file names to cam.txt NOT cam.txt.txt

## Instructions for Running EndoSlam Preprocessing
1) Either use parameters in endoslam_proprocess.py or use prepare_endoslam.sh. Will need larger GPUs if using the bash script.
2) Once all data has been preprocessed, run train_test_split.py with required arguments.
3) endoslam_dataset.py contains custom Dataset class to read in the train_test splits for Monst3R
4) visualiize.py allows you to visualize the preprocessed camera frames and associated depth maps. 
