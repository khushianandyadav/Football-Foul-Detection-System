# Football Foul Detection System

AI-powered Football Foul Detection using YOLOv8, MVFoul, OpenCV and Streamlit.

## Features

- Upload football videos
- Live camera analysis
- Player detection using YOLOv8
- Foul recognition using MVFoul
- Automatic foul snapshots
- Annotated video generation
- Downloadable results

## Technologies

- Python
- Streamlit
- OpenCV
- YOLOv8
- MVFoul
- PyTorch

## Installation

```bash
git clone https://github.com/khushianandyadav/Football-Foul-Detection-System.git

cd Football-Foul-Detection-System

pip install -r requirements.txt

streamlit run app.py
```

## Pretrained Models

The pretrained model weights are not included in this repository because they exceed GitHub's file size limit.

Download the pretrained models from the Google Drive folder below:

 **Google Drive:**  
https://drive.google.com/drive/folders/1dxRGn_wDdTrDIo1ZWHPf93JotMPreNk_?usp=drive_link

After downloading, place the files in the project root directory:

```
Football-Foul-Detection-System/
│
├── 14_model.pth.tar
├── sport_classifier.pth
├── yolov8n.pt
│
├── app.py
├── frame_extractor.py
└── ...
```

Required model files:

- `14_model.pth.tar`
- `sport_classifier.pth`
- `yolov8n.pt`
