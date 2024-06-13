Keyboard layout automation check

Author: AI application engineer intern Benson Lin

Project Describe: Improve traditional detection method with YOLO V8 model, then generate the excel report, this report records the detailed results of the automation check.

Python file name (with YOLO and report generate): LayoutChecker_v20240613_YOLO.py

Python file name (NO YOLO): LayoutChecker_v20240613_no_YOLO.py

Python Version: 3.8.18

How to execute project: 
Step1: pip install - r requirements.txt

Step2: Drag the compare file into folder(Design, PDK or Reference), for example: Drag “Chengdu Universal US INT'L V3.3.pdf” and “Chengdu Universal US INT'L V4.0.pdf” into design folder.

Step3: python LayoutChecker_v20240613_YOLO.py

Step4: After the inference, you can get the results like bellow Fig 1 and Fig 2:
