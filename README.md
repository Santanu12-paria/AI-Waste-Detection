# 🗑️ AI Based Waste Detection

A computer vision project that detects and classifies waste in images using **YOLOv8**, with a **Streamlit** web app.

## Features
- Upload a JPG/PNG image and detect waste objects with bounding boxes
- Class name and confidence for every detection
- Total objects, waste types and average confidence
- Waste level (LOW / MEDIUM / HIGH)
- Adjustable confidence threshold

## Waste classes (11)
Aluminum can, Cardboard, Paper, Container for household chemicals, Glass bottle, Organic, Plastic bag, Plastic bottle, Plastic cup, Printing industry, Tin

## Model and results
- Model: YOLOv8m, 640 px, up to 80 epochs with early stopping, trained on Google Colab (T4 GPU)
- Validation mAP50: **0.789** (mAP50-95: 0.587)
- Test mAP50: **0.761**

| Class | mAP50 (val) |
|---|---:|
| Printing industry | 0.98 |
| Aluminum can | 0.94 |
| Glass bottle | 0.90 |
| Plastic cup | 0.88 |
| Paper | 0.84 |
| Cardboard | 0.78 |
| Organic | 0.76 |
| Tin | 0.71 |
| Plastic bottle | 0.69 |
| Plastic bag | 0.62 |
| Container for household chemicals | 0.58 |

## Dataset
Roboflow Universe "YOLO Waste Detection" (project-fpxlk/yolo-waste-detection-mroxj, CC BY 4.0), 5,460 images in 42 classes.
Most classes had too few examples to learn, so classes with fewer than 100 training objects were removed and a few small
ones were merged (Zip plastic bag into Plastic bag; Paper bag, Paper shavings and Cellulose into Paper). This left 11 classes.

## Run locally
```
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```
The trained weights (`best.pt`) are not stored in the repository. Download them from the Releases page and place the file at `models/best.pt`.

## Limitations
The model works best on clear, separated objects. It struggles with dense piles of overlapping waste:
- Most items in a heap are missed, and sometimes the whole heap is boxed as one object.
- Training images contain only 1 to 2 labeled objects each on average, so crowded scenes are outside what the model learned.
- The waste level is based on the number of detected objects, so a large pile with few detections can be reported as LOW.
- Weakest classes: Plastic bag, Plastic bottle, Tin and Container for household chemicals.

## Future work
- Add labeled photos of dense, cluttered scenes
- Add more examples for the weak classes
- Real-time camera detection
- Online deployment

## License
For educational use. The dataset follows its own CC BY 4.0 license.