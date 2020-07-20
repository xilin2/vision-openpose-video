## pytorch-openpose + yolo_hand_detection for experiment 2

A combination of two GitHub repositories: [Pytorch OpenPose](https://github.com/Hzzone/pytorch-openpose) and [Yolo Hand Detection](https://github.com/cansik/yolo-hand-detection)

The two repos above have pre-existing connections between files (via imports, etc.). I had to preserve some of these connections (or at least to my knowledge) since some of these connections were encoded in files which I was unable to open through a text editor. To bypass this, the files from the two repos have been organized in a very specific and slightly confusing way. I provide instructions on how to donwload and organize these files.

### Getting Started

Assuming you already have Pytorch-Openpose downloaded and working correctly, 

**1. Create a new local project directory.**

**2. From your original Pytorch-Openpose directory, copy over the folders "src," "notebooks," and "model" into your new directory** (I think this would be easier than re-downloading and compiling all of Pytorch-Openpose)

**3. In the folder "src," delete util.py.** This will be replaced by the util.py included in this repo.

**4. Clone a copy of Yolo-Hand-Detection inside your new directory.** You should now have the folders, "src," "notebooks," "model," and "Yolo-Hand-Detection" in your project directory.
    
    git clone https://github.com/cansik/yolo-hand-detection.git
    
**5. In the new "Yolo-Hand-Detection" directory, install requirements using the terminal code below.**
    
    pip install -r requirements.txt
    
**6. Then, download the models and weights in the same directory.**
   
    sh models/download-models.sh 
    
**7. Copy the experiment videos into a new folder in "Yolo-Hand-Detection"** (I used the name "Yolo-Hand-Detection/exp/vids_set" which is the path I used in the code as well) 

**8. From here, you can download the files listed in this repo. Each file will be placed as such:**

File | Location
------------ | -------------
vid_exp2.py | Yolo-Hand-Detection
detect_hands.py | Yolo-Hand-Detection
hand_vid.py | src _(openpose)_
util.py | src _(openpose)_

### Run the Program

1. Open vid_exp2.py. **Check to make sure that the program will be analyzing and outputting to the correct video directory.** The code can be found at the bottom.
    ```python
        if __name__ == "__main__":
            dir = '[DIRECTORY]/' # I use 'exp/vids_set/'
            ...
            scipy.io.savemat('[DIRECTORY]/set1_body.mat', b_data)
            scipy.io.savemat('[DIRECTORY]/set1_hand.mat', h_data)
    ```
2. **Navigate to "Yolo-Hand-Detection" in terminal.** This is where the code will be run.

3. **Run.**
    ```bash
    python video_exp2.py
    ```
