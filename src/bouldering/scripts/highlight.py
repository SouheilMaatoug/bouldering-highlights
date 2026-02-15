import argparse
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from src.bouldering.features.audio import (
    apply_filter,
    delta_signal,
    resample_waveform,
    rms_energy,
    zscore_signal,
)
from src.bouldering.features.audio_semantics import crowd_prediction
from src.bouldering.features.events import detect_crux, detect_dyno, detect_fall, detect_top
from src.bouldering.features.motion import time_derivative
from src.bouldering.features.pose import cog_y, hands_above_shoulders, normalize_cog_y, pose_visibility_ratio
from src.bouldering.features.temporal import displacement_amplitude, sliding_window
from src.bouldering.media.video.video import Video
from src.bouldering.models.audio.yamnet import YamNetClassifier
from src.bouldering.models.detection.yolo import YoloPersonDetector
from src.bouldering.models.ocr.splitter import SceneSplitterOCR
from src.bouldering.models.pose.pose import MediaPipePoseEstimator
from src.bouldering.scoring.audio import compute_audio_score
from src.bouldering.scoring.events import apply_event_boosts
from src.bouldering.scoring.score import compute_overall_score
from src.bouldering.scoring.utils import (
    detect_peaks,
    ema_smooth,
    extract_segments,
    fill_none_with_zero,
    interpolate_signal,
)
from src.bouldering.scoring.visual import compute_visual_score
from src.bouldering.structure.scenes.content import ContentSplitter
from src.bouldering.structure.scenes.ocr import OCRSplitter
from src.bouldering.structure.scenes.pipelines import ScenePipeline
from src.bouldering.structure.segments.segments import merge_segments
from src.bouldering.tracking.activity import detect_active_competitors
from src.bouldering.tracking.postprocessing import merge_tracks
from src.bouldering.tracking.tracking import YoloPersonTracker

ALPHA = 0.65
BETA = 0.35

EVENT_PARAMS = {
    "DYNO": {"tau": 0.35, "max_boost": 0.6},
    "FALL": {"tau": 0.35, "max_boost": 0.4},
    "CRUX": {"tau": 0.8, "max_boost": 0.3},
    "TOP": {"tau": 1.0, "max_boost": 0.8},
}


def split_scenes(video: Video):
    """Split a video into Boulders and scenes."""
    # OCR splitter
    ocr_logic = SceneSplitterOCR(
        langs=["en"],
        crop_box=[0, 0.5, 0.5, 1],  # <-- relative coords
        fx=0.5,
        fy=0.5,
        stride=3,
        batch_size=16,
        smooth_window=5,
        majority_ratio=0.6,
        require_number=True,
    )

    ocr_splitter = OCRSplitter(ocr_logic)

    # Content splitter
    content_splitter = ContentSplitter(
        threshold=27.0,
        min_scene_len_sec=1.5,
        downscale=2,
    )

    # Wrapper pipeline
    pipeline = ScenePipeline(
        macro_splitter=ocr_splitter,
        micro_splitter=content_splitter,
    )

    # run the pipeline
    scenes = pipeline.run(video)
    return scenes


def aggregate_visual_score_signals(
    visual_scores: List[List[Tuple[float, float]]],
    second_weight: float = 0.6,
) -> List[Tuple[float, float]]:
    """Aggregate multiple per-competitor visual score signals into a single score.

    At each time step, the aggregated score is computed as a capped sum of
    the top-2 visual scores:
        score(t) = clamp(v1 + second_weight * v2)

    This allows multiple active climbers to contribute while preventing
    score explosion.

    Args:
        visual_scores: List of visual score signals, one per competitor.
            Each signal is a list of (time, score) tuples.
            All signals must have the same length and time base.
        second_weight: Contribution weight for the second most active
            competitor.

    Returns:
        Aggregated visual score signal as a list of (time, score).

    Raises:
        ValueError: If input signals are empty or misaligned.
    """
    if not visual_scores:
        return []

    n = len(visual_scores[0])

    for vs in visual_scores:
        if len(vs) != n:
            raise ValueError("All visual score signals must have the same length")

    aggregated = []

    for i in range(n):
        t = visual_scores[0][i][0]

        scores_at_t = [vs[i][1] for vs in visual_scores if vs[i][1] is not None]

        if not scores_at_t:
            aggregated.append((t, 0.0))
            continue

        scores_at_t.sort(reverse=True)

        v1 = scores_at_t[0]
        v2 = scores_at_t[1] if len(scores_at_t) > 1 else 0.0

        score = v1 + second_weight * v2
        aggregated.append((t, min(1.0, score)))

    return aggregated


def main():
    """Main pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-video", required=True, help="Video to read.")
    parser.add_argument("--output-video", required=True, default="./highlights.mp4", help="Video to write.")
    args = parser.parse_args()

    # 1. read the input video
    video = Video.read(args.input_video)
    fps = video.sequence.fps
    frame_shape = video.sequence.frame(0).shape
    H = frame_shape[0]
    overall_segments = []
    # 2. split into scenes
    print("Splitting the video")
    scenes = split_scenes(video)

    # Loop over all detected scenes:
    for scene in tqdm(scenes):
        # get the scene video
        video_part = video.cut(scene.start_time, scene.end_time)
        sequence = video_part.sequence
        if sequence.n_frames < 100:
            continue
        # time array
        t = [i / fps for i in range(sequence.n_frames)]
        # Compute features and scores

        # 1. Audio score
        # audio features (scene level)
        audio = video_part.audio.samples  # raw waveform (mono or stereo)
        sr = video_part.audio.sample_rate  # original sampling rate
        # yamnet prediction
        model = YamNetClassifier()
        waveform_16k = resample_waveform(audio, sr, 16000)
        yamnet_output = model.predict(waveform_16k, 16000)
        crowd_score = crowd_prediction(
            yamnet_output,
            target_classes=("Crowd", "Cheering", "Shout"),
        )
        # filter audio
        window = 0.48  # 400 ms
        f = 500  # Hz
        filtered_audio = apply_filter(audio, sr, filter_type="highpass", low_hz=f)
        rms = rms_energy(filtered_audio, sr, window_seconds=window)
        delta_rms = delta_signal(rms)
        z_rms = zscore_signal(rms)
        # interpolate
        delta_rms_interp = interpolate_signal(delta_rms, t)
        z_rms_interp = interpolate_signal(z_rms, t)
        crowd_interp = interpolate_signal(crowd_score, t)
        audio_score = compute_audio_score(
            delta_rms_signal=delta_rms_interp,
            z_rms_signal=z_rms_interp,
            crowd_signal=crowd_interp,
        )
        # 2. Visual score
        visual_scores = []
        # detect competitors
        model = YoloPersonDetector()
        tracker = YoloPersonTracker(model, fps=fps)
        tracks = tracker.track_frames(sequence.frames())
        # postprocess tracks
        merged_tracks = merge_tracks(tracks, frame_shape)
        # for each tracked person, detect the active competitors
        active_tracks = detect_active_competitors(
            tracks=merged_tracks,
            frame_shape=video.sequence.frame(0).shape,
            fps=video.sequence.fps,
            window_seconds=5.0,
            min_motion=0.002,
            min_presence=0.3,
        )
        # for each detected track:
        for track in active_tracks.values():
            # pose estimation
            pose_estimator = MediaPipePoseEstimator()
            pose_track = []
            for entry in track:
                frame = sequence.frame(entry["frame"])
                bbox = entry["bbox"].as_xyxy()
                landmarks = pose_estimator.estimate(frame, bbox)
                if landmarks is None:
                    continue
                pose_track.append(
                    {
                        "frame": entry["frame"],
                        "time": entry["frame"] / fps,
                        "landmarks": landmarks,
                    }
                )
            if len(pose_track) < 100:
                continue
            # visual features
            # build core feature signals
            # CoG Y
            cog_y_norm = [(f["time"], normalize_cog_y(cog_y(f), H)) for f in pose_track]
            # CoG velocity
            cog_velocity_norm = time_derivative(cog_y_norm)
            # CoG acceleration
            cog_acceleration_norm = time_derivative(cog_velocity_norm)
            # Pose visibility
            visibility = [(f["time"], pose_visibility_ratio(f)) for f in pose_track]
            # Hands above head
            hands_up = [(f["time"], hands_above_shoulders(f)) for f in pose_track]
            # event detection:
            # fall detection
            falls = detect_fall(
                velocity_norm=cog_velocity_norm,
                cog_y_norm=cog_y_norm,
                visibility=visibility,
                velocity_threshold=-0.3,
                ground_threshold=0.3,
                min_duration=0.2,
            )
            # dyno detection
            # Displacement amplitude (normalized, 0–1)
            windows = sliding_window(
                [(t, y) for t, y in cog_y_norm if y is not None],
                window_seconds=0.5,
            )
            displacement_amplitude_norm = [(w[-1][0], displacement_amplitude(w)) for w in windows if len(w) > 1]
            dynos = detect_dyno(
                velocity_norm=cog_velocity_norm,
                acceleration_norm=cog_acceleration_norm,
                displacement_amplitude_norm=displacement_amplitude_norm,
                min_velocity=0.25,  # strong upward motion
                min_acceleration=2.0,  # explosive
                min_amplitude=0.08,  # large vertical displacement
                max_duration=0.6,
            )
            # crux detection
            # Motion energy proxy = |velocity|
            # Motion variance (from velocity)
            motion_windows = sliding_window(
                [(t, v) for t, v in cog_velocity_norm if v is not None],
                window_seconds=2.0,
            )
            motion_variance = [(w[-1][0], np.var([v for _, v in w])) for w in motion_windows if len(w) > 1]
            # Vertical progress (from CoG)
            progress_windows = sliding_window(
                [(t, y) for t, y in cog_y_norm if y is not None],
                window_seconds=2.0,
            )
            vertical_progress = [
                (w[-1][0], max(y for _, y in w) - min(y for _, y in w)) for w in progress_windows if len(w) > 1
            ]
            # crux detection
            cruxes = detect_crux(
                motion_variance=motion_variance,
                vertical_progress=vertical_progress,
                min_duration=2.5,
                max_progress=0.15,
            )
            # detect tops
            motion_energy = [(t, abs(v)) for t, v in cog_velocity_norm if v is not None]
            tops = detect_top(
                hands_above_shoulders=hands_up,
                cog_y_norm=cog_y_norm,
                motion_energy=motion_energy,
                top_threshold=0.75,  # near top
                max_motion=0.08,  # stable
                min_duration=0.1,
            )
            # visual score
            cog_velocity_clean = fill_none_with_zero(cog_velocity_norm)
            displacement_clean = fill_none_with_zero(displacement_amplitude_norm)
            cog_velocity_interp = interpolate_signal(cog_velocity_clean, t)
            displacement_interp = interpolate_signal(displacement_clean, t)
            visual_score = compute_visual_score(
                v_y_signal=cog_velocity_interp,
                displacement_signal=displacement_interp,
                weights=(0.3, 0.5, 0.2),
            )
            # add event boost
            all_events = []
            all_events.extend(falls)
            all_events.extend(dynos)
            all_events.extend(cruxes)
            all_events.extend(tops)
            visual_boost_score = apply_event_boosts(
                visual_score,
                all_events,
                EVENT_PARAMS,
            )
            visual_scores.append(visual_boost_score)
        visual_score = aggregate_visual_score_signals(visual_scores)
        # overall score: audio + visuals
        overall_score = compute_overall_score(audio_score, visual_score, ALPHA, BETA)
        # smooth score
        final_score_smooth = ema_smooth(overall_score, alpha=0.25)
        # detect peaks
        peaks = detect_peaks(
            final_score_smooth,
            min_height=0.45,
            min_distance=1.0,
        )
        # extract segments
        segments = extract_segments(peaks, pre=1.5, post=2.5)
        merged_segments = merge_segments(segments)
        for segment in merged_segments:
            segment["start"] += scene.start_time
            segment["end"] += scene.start_time
        # add scene start to segments
        # append to global segments
        overall_segments.extend(merged_segments)
    overall_segments_sorted = sorted(overall_segments, key=lambda s: s["start"])
    overall_segments_merged = merge_segments(overall_segments_sorted)
    parts = [video.cut(s["start"], s["end"]) for s in overall_segments_merged]
    video_highlights = Video.concatenate(parts)
    video_highlights.write(args.output_video)
    print("done")


if __name__ == "__main__":
    main()
