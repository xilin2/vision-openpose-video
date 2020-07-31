## pytorch-openpose + yolo_hand_detection for experiment 2

A combination of two GitHub repositories: [Pytorch OpenPose](https://github.com/Hzzone/pytorch-openpose) and [Yolo Hand Detection](https://github.com/cansik/yolo-hand-detection). **Requires Pytorch.**

### Clone repo

    git clone https://github.com/xilinzhou/vision-openpose-video.git
    
### Install requirements
While in project root directory, install other requirements. 

    pip install -r requirements.txt

### Add Models

There are two sets of models to be added: Pytorch OpenPose and Yolo Hand Detection

**1. Add Pytorch OpenPose models.**
You have two options. If you have the models already available, you can copy over the models into the "model" folder found in the project root directory. Otherwise, you can download the models into the same "model" folder mentioned previously using [Dropbox](https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABWFksdlgOMXR_r5v3RwKRYa?dl=0). After doing so, you should have the following files in your "model" folder.

* body_pose.caffemodel
* hand_pose.caffemodel
* body_pose_deploy.prototxt
* hand_pose_deploy.prototxt
* body_pose_model.pth
* hand_pose_model.pth

**2. Add Yolo Hand Detections models**
Again, you have two options. If already downloaded, you can copy over the models into your "Yolo-Hand-Detection/models" folder. Otherwise, you can download the models in terminal while in the "Yolo-Hand-Detection" directory.
    
    cd Yolo-Hand-Detection
    sh models/download-models.sh 
    
Once complete, you should have the following files in your "Yolo-Hand-Detection/models" folder. (You may have to move these files into the correct folder)

* cross-hands-tiny-prn.cfg
* cross-hands.cfg
* cross-hands-tiny-prn.weights
* cross-hands.weights
* cross-hands-tiny.cfg
* download-models.sh
* cross-hands-tiny.weights

### Run Data Extraction

1. Open *video_exp2.py* found in the "Yolo-Hand-Detection" folder. **Check to make sure that the program will be analyzing and outputting to the correct video directory.** The lines to do so can be found at the code's bottom.
    ```python
    if __name__ == "__main__":
        dir = '[DIRECTORY]/'
    ```
2. **Navigate to "Yolo-Hand-Detection" in terminal.** This is where the code will be run.

3. **Run.** This will output two .mat files storing data for hand and body features. 
    ```bash
    python video_exp2.py
    ```

### Run Data Analysis

1. Open *analysis.py* found in project root directory. **Make sure that the correct paths for the feature data .mat files and the video name key are listed.** The lines to do so can be found at the start of the Python Main function. 
    ```python
    if __name__ == "__main__":
        body = scipy.io.loadmat('[PATH]/vids_set_body.mat')
        hand_box = scipy.io.loadmat('[PATH]/vids_set_hand.mat')
        hand_pose = scipy.io.loadmat('[PATH]/set1_hand.mat')
        xls = pd.ExcelFile('[PATH]/vidnamekey.xlsx')
    ```
    
2. **Run.** This will save several dendrogram and heatmaps as .pngs found in the "plots" folder, as well as print in terminal the correlation values between the features of video sets 1 and 2.
    ```bash
    python analysis.py
    ```
    
    
