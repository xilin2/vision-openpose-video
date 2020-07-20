## pytorch-openpose + yolo_hand_detection for experiment 2

A combination of two GitHub repositories: [Pytorch OpenPose](https://github.com/Hzzone/pytorch-openpose) and [Yolo Hand Detection](https://github.com/cansik/yolo-hand-detection)

I wasn't quite sure for now how to get all the files linked up, so I provided some instructions on how to organize the files.

### Getting Started

Assuming you already have the necessary requirements downloaded for Pytorch-Openpose, 

1. Create a new local project directory.

2. From your original Pytorch-Openpose directory, copy over the folders "src," "notebooks," and "model." (I think this would be easier than re-downloading and compiling all of Pytorch-Openpose)

3. In the folder "src," delete util.py. This will be replaced by the util.py included in this repo.

4. Clone a copy of Yolo-Hand-Detection inside your new directory.

    ```bash
    git clone https://github.com/cansik/yolo-hand-detection.git
    ```
5. In the new "Yolo-Hand-Detection" directory, install requirements.
    ```bash
    pip install -r requirements.txt
    ```
6. Then, download the models and weight in the same directory using terminal.

    ```bash
    sh models/download-models.sh 
    ```
7. Copy the experiment videos into the "exp" folder found in "Yolo-Hand-Detection"

8. From here, you can download the files listed in this repo. Each file will be placed as such:

File | Location
------------ | -------------
vid_exp2.py | Yolo-Hand-Detection
detect_hands.py | Yolo-Hand-Detection
hand_vid.py | src
util.py | src

### Run the Program

1. Open vid_exp2.py. Check to make sure that the program will be analyzing and outputting to the correct video directory. The code can be found at the bottom.
    ```python
        if __name__ == "__main__":
        dir = 'exp/[DIRECTORY]/'
        b_data, h_data = get_vid_data(dir) # Dictionaries with keys=filenames, each storing 2d numpy array with shapes (75,72), (75,126)
        scipy.io.savemat('exp/[DIRECTORY]/set1_body.mat', b_data)
        scipy.io.savemat('exp/[DIRECTORY]/set1_hand.mat', h_data)
    ```
2. Navigate to "Yolo-Hand-Detection" in terminal. This is where the code will be run.

3. Run.
    ```bash
    python video_exp2.py
    ```
