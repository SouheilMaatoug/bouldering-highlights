# bouldering-highlights

An **end-to-end AI pipeline** to automatically create **highlights** from Bouldering competition videos.

This project combines:
- **Computer vision** (Person detection, tracking, pose estimation)
- **OCR-based segmentation**
- **Audio event detection** (audio signal processing + Classification using YAMnet)
- **Visual Feature extraction** 
- **Heuristic event rules**
- **Replay detection (optional)**
- **Event scoring & selection**


## 📦 Features

- Detects Bouldering sections (“Boulder n”) using scene detection + OCR
- Tracks all persons, identifies the **active climber**
- Extracts pose-based kinematic features
- Analyzes the audio track for applause/cheering/shouts
- Detects key events:
  - attempts  
  - dynamic moves (crux)  
  - falls  
  - tops  
- Handles replays (optional)
- Scores events and selects the best per section
- Generates:
  - **events.json**
  - **highlights.mp4**

---

# 🧭 Pipeline Overview
```mermaid
flowchart TD
    A[Input Video] --> B[Preprocessing, extraction<br/>frames,  metadata, audio];
    B --> C[Section Segmentation<br/>scene detection + OCR Boulder n];
    C --> D[Vision Analysis<br/>YOLO -> person detection];
    D --> E[Active Climber Identification - crop];
    E --> F[Visual features<br/>MediaPipe -> Pose estimation]
    C --> G[Audio Analysis<br/>RMS + YAMNet];
    D --> G[Audio Features];
    G --> H[Event Detection<br/>attempt / crux / fall / top];
    F --> H;
    H --> J[Event Scoring];
    J --> K[Event Selection per Section];
    K --> L[Highlights Video];
```
