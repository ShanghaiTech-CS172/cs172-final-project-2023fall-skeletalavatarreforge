### To run Easymocap

# Optional 
# Save your own video in the videos directory 
# data/own_data
#   ├── videos

#Set up environment

# cd EasyMocap
# conda create -n easymocap python=3.9 -y
# conda activate easymocap
# wget -c https://download.pytorch.org/whl/cu116/torch-1.12.0%2Bcu116-cp39-cp39-linux_x86_64.whl
# python3 -m pip install ./torch-1.12.0+cu116-cp39-cp39-linux_x86_64.whl
# wget -c https://download.pytorch.org/whl/cu116/torchvision-0.13.0%2Bcu116-cp39-cp39-linux_x86_64.whl
# python3 -m pip install ./torchvision-0.13.0+cu116-cp39-cp39-linux_x86_64.whl
# python -m pip install -r requirements.txt
# pip install spconv-cu116
# pip install pytube
# # install pyrender if you have a screen
# python3 -m pip install pyrender
# python setup.py develop

data=data/own_data

# Optional: Download the video from Youtube

# python3 scripts/dataset/download_youtube.py "https://www.youtube.com/watch?v=DWnxgtEGTqw" --database ${data}

#Rename your input video or you need to change --subs to your video name

# cd ${data}/videos

# FILE=$(find './' -type f \( -name "*.mp4" -o -name "*.mov" -o -name "*.webm" \) | head -n 1)

# if [ -z "$FILE" ]; then
#     echo "wrong"
#     exit 1
# else
#     echo "1"
#     ffmpeg -i "$FILE" -c copy input.mp4
#     if [[ "$FILE" != "./input.mp4" ]]; then
#         rm "$FILE"
#     fi
# fi
# cd ..
# cd ..
# cd ..

#Extract KeyPoints --ext means your image type set up the openpose environment if you want to run the code involving openpose

# python3 apps/preprocess/extract_keypoints.py ${data} --mode yolo-hrnet
# python3 apps/preprocess/extract_keypoints.py ${data} --mode openpose --openpose ./openpose --hand --face
python3 apps/preprocess/extract_keypoints.py ${data} --mode openpose --openpose openpose --ext .png

# DON'T FORGET TO CHANGE YOUR PICTURE TYPE IN svimage.yml IF YOU USE ORIGINAL IMAGES
emc --data config/datasets/svimage.yml --exp config/1v1p/owndata.yml --root ${data} --ranges 0 50 1 --subs input --out output/own_data1s

# cd ..
