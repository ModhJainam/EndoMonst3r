## Download and Formatting Instructions for EndoSlam
wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/cd2rtzm23r-1.zip

unzip "*.zip" all zipped files inside folders

Restructure the folders so that the Cameras folder contains HighCam, LowCam, MiroCam, and UnityCam only -> Remove OlympusCam and PillCam, they cannot be used as there is either no pose information or no ground truth.

Edit Camera organ folders to match names in 3D Scanner folders (e.g. Change Colon-III(L-shaped) to Colon-III)
Edit HighCam and LowCam calibration file names to cam.txt NOT cam.txt.txt

