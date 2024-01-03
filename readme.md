# Enhanced Proctoring System: Computer Vision

## Overview
The Enhanced Proctoring System is a computer vision-based solution designed for monitoring and enhancing security during educational assessments. The system employs various computer vision techniques to detect and track facial features, eye movements, and gaze direction for a more robust and accurate proctoring experience.

## Important Notes
- Ensure that the required dependencies are installed using the provided `requirements.txt`.
- Camera calibration is crucial for accurate tracking; Once you run the main.py follow the instructions in the background.
- Pre-trained model weights (`best.pt`) and additional resources can be found in respective directories.
- Before running the software make sure to add the photo of the user in `FaceEncode/Images` so that the system can give you the access to take test.
- Incase of testing the software make sure to add the video of the user in `FaceEncode/Video`, aslo change the video directory in the `main_test.py` file, 11th line of the code to the newly added video.

## Personalization Regarding Deceptive Practices
The thresholds for different types of fraud can be easily customized in the Enhanced Proctoring System. This customization is especially helpful for adjusting the system to accommodate student behavior variations during longer exams and varying test durations.
- Parameters for Configuration:
  - The professor (you) can modify the following configuration parameters to determine the thresholds for fraudulent activity:
    - **`max_eye_fault`:**
     This parameter represents the maximum allowable count of eye-related faults before considering it as a potential fraudulent activity. Adjust this value based on the acceptable tolerance for eye-related movements during the test.

    - **`max_eye_warning`:**
     Sets the maximum number of warnings issued for eye-related behaviors. Once this threshold is reached, the system may raise alerts or take predefined actions to notify potential issues.

    - **`max_suspicious_count`:**
     Defines the threshold for the maximum count of overall suspicious activities. If this threshold is exceeded, it indicates a higher likelihood of fraudulent behavior.

    - **`mouth_thres`:**
     Represents the threshold for mouth-related movements. Adjust this value based on the acceptable tolerance for mouth-related behaviors during the test.

  - Example Configuration:
    ```python
    # Set the threshold for the various fraudulent activities
    max_eye_fault = 10
    max_eye_warning = 1
    max_suspicious_count = 6
    mouth_thres = 4
    ```

## Prerequisites
Use these instructions to launch the programs in this folder:
(Below commands are for the Mac users only)

- Before running the Enhanced Proctoring System, make sure you have Python 3.8 installed. If you don't have it installed, you can download it from the [official Python website](https://www.python.org/downloads/).
- Use the following command to install the required dependencies using pip:
```bash
  pip install -r requirements.txt
```
- Use the below command to start a virtual environment:
  - `python -m venv proctoring_venv`
- Activate the virtual environment proctoring_venv that you have previously installed
  - `source /proctoring_venv/bin/activate`
- Install the requirements into the newly created virutal environments.
  - `pip install -r requirements.txt`

## Usage
- Execute the main script (`main.py`) to start the Enhanced Proctoring System, you can use the below commands to run:
  - `python main.py`

## Testing
- Utilize the `main_test.py` script to test individual video of the system, you can use the below commands to run:
  - `python main_test.py`

## Project Structure
- __pycache__: Python cache files (automatically generated)
- best.pt: Pre-trained model weights or checkpoints
- camera_calibration.py: Script for camera calibration
- eye_motion_tracking.py: Module for eye motion tracking
- FaceEncode: Directory for face encoding models or data
  - __pycache__: Python cache files for face_encode module
  - face_encode.cpython-38.pyc: Compiled Python file for face_encode module
  - face_encode.py: Module for face encoding
  - FraudImage: Directory for images related to potential fraud detected image stored.
  - Images: Directory for storing images relevant to face encoding
  - Result: Directory to store results or outputs
  - Video: Directory for video files related to face encoding
- main_test.py: Test script for validating system components.
- main.py: Main script for running the Enhanced Proctoring System
- mouth.py: Module for mouth movement tracking
- proctor_venv: Virtual environment for the proctoring module (example virtual environment directory)
- requirements.txt: List of project dependencies
- Resources: Directory for additional resources (images)
- setup_calibration.py: Script for setting up camera calibration
- test.py: Script for running mobile detection tests
- track_gaze.py: Module for gaze tracking
- YOLOv8 Mobile Detection: YOLOv8 implementation for object detection

## Resources
- Add any additional documentation, guides, or images in the 'Resources' directory.

## Calibration

![result1](https://github.com/nitinshivakumar/Enhanced-Proctoring-System-Computer-Vision/blob/main/FaceEncode/Result/2024-01-03%2003%3A02%3A28%20%2B0000.GIF)

## Results

![result1](https://github.com/nitinshivakumar/Enhanced-Proctoring-System-Computer-Vision/blob/main/FaceEncode/Result/2024-01-01%2023%3A16%3A17%20%2B0000.GIF)

![result2](https://github.com/nitinshivakumar/Enhanced-Proctoring-System-Computer-Vision/blob/main/FaceEncode/Result/GMP_U2F2ZUdIMDE%3D%202.GIF)

![result3](https://github.com/nitinshivakumar/Enhanced-Proctoring-System-Computer-Vision/blob/main/FaceEncode/Result/GMP_U2F2ZUdIMDE%3D.GIF)

![result4](https://github.com/nitinshivakumar/Enhanced-Proctoring-System-Computer-Vision/blob/main/FaceEncode/Result/2024-01-03%2002%3A58%3A12%20%2B0000.GIF)