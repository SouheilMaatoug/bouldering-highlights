# bouldering-highlights

An **end-to-end AI pipeline** to automatically create **highlights** 
from the Bouldering Olympics 2024 competition videos.

![bouldering](images/bouldering_resized.png)

This project intergrates pretrained Computer Vision models such as **YOLO** for person detection for example, and Image Processing techniques with rule based logic to identify key events.

For this first version of the project, no model training or fine-tuning is performed. Person detection or pose estimations are already well knwon and developed fields and pretrained models are sufficiently performant and robust.

System rules and numeric parameters are derived from an example video.
However, as with any ML-based system, an evaluation phase is essential to validate the performance and understand the system's behavior. Custom metrics will be defined and a dedicated test video will serve as the evaluation benchmark.

Technology used:
- **Video / audio**
  - FFMPEG (read/write)
- **Visual analysis**:
  - Person detection (YOLOv11)
  - OCR-based segmentation (EasyOCR)
  - Pose estimation (MediaPipe)
- **Audio analysis**
  - sound classification (YAMNet)

---

# Highlights pipeline
```mermaid
flowchart TD
    A[Input Video] --> B[Preprocessing, extraction<br/>frames,  metadata, audio];
    B --> C[Section Segmentation<br/>scene detection + OCR Boulder n];
    
    C --> D[Vision Analysis<br/>YOLO -> person detection];
    D --> E[Active Climber Identification - crop];
    E --> F[Visual features<br/>MediaPipe -> Pose estimation]

    C --> G[Audio Analysis<br/>RMS + YAMNet];
    G --> H[Audio Features];
    
    H --> I[Event Detection<br/>attempt / crux / fall / top];
    F --> I;
    I --> J[Event Scoring];
    J --> K[Event Selection per Section];
    K --> L[Highlights Video];
```
