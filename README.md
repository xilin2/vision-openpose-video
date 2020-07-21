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
You have two options. If you have the models already available, you can copy over the models into the directory "model" found in the project root directory. Otherwise, you can download the models into the same "model" folder from [Dropbox](https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABWFksdlgOMXR_r5v3RwKRYa?dl=0). You should have the following files in your "model" folder.

* body_pose.caffemodel
* hand_pose.caffemodel
* body_pose_deploy.prototxt
* hand_pose_deploy.prototxt
* body_pose_model.pth
* hand_pose_model.pth

**2. Add Yolo Hand Detections models**
Again, you have two options. If already downloaded, you can copy over the models into your "Yolo-Hand-Detection/models" directory. Otherwise, you can download the models in terminal.

    sh models/download-models.sh 
    
Once complete, you should have the following files in your "Yolo-Hand-Detection/models" folder.

* cross-hands-tiny-prn.cfg
* cross-hands.cfg
* cross-hands-tiny-prn.weights
* cross-hands.weights
* cross-hands-tiny.cfg
* download-models.sh
* cross-hands-tiny.weights

### Run the Program

1. Open vid_exp2.py. **Check to make sure that the program will be analyzing and outputting to the correct video directory.** The code can be found at the bottom.
    ```python
    if __name__ == "__main__":
        dir = '[DIRECTORY]/'
    ```
2. **Navigate to "Yolo-Hand-Detection" in terminal.** This is where the code will be run.

3. **Run.**
    ```bash
    python video_exp2.py
    ```
