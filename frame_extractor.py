import os
import cv2

# -----------------------------
# CONFIGURATION
# -----------------------------

VIDEO_ROOT = "videos"
SOCCER_VIDEO_FOLDER = os.path.join(VIDEO_ROOT, "soccer")
NONSOCCER_VIDEO_FOLDER = os.path.join(VIDEO_ROOT, "nonsoccer")

DATASET_ROOT = "dataset"
SOCCER_OUTPUT = os.path.join(DATASET_ROOT, "soccer")
NONSOCCER_OUTPUT = os.path.join(DATASET_ROOT, "nonsoccer")

FRAME_INTERVAL = 20           # Save every 20th frame
MAX_FRAMES_PER_VIDEO = 250    # Safety cap

# -----------------------------
# CREATE OUTPUT FOLDERS
# -----------------------------

os.makedirs(SOCCER_OUTPUT, exist_ok=True)
os.makedirs(NONSOCCER_OUTPUT, exist_ok=True)

# -----------------------------
# FRAME EXTRACTION FUNCTION
# -----------------------------

def extract_from_folder(input_folder, output_folder):

    if not os.path.exists(input_folder):
        print(f"❌ Folder not found: {input_folder}")
        return

    video_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith((".mp4", ".avi", ".mov"))]

    if len(video_files) == 0:
        print(f"⚠️ No videos found in {input_folder}")
        return

    for video_file in video_files:

        video_path = os.path.join(input_folder, video_file)
        print(f"\nProcessing: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("❌ Could not open video.")
            continue

        frame_count = 0
        saved_count = 0
        video_name = os.path.splitext(video_file)[0]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_INTERVAL == 0:
                filename = f"{video_name}_{saved_count}.jpg"
                save_path = os.path.join(output_folder, filename)

                cv2.imwrite(save_path, frame)
                saved_count += 1

                if saved_count >= MAX_FRAMES_PER_VIDEO:
                    break

            frame_count += 1

        cap.release()
        print(f"✅ Saved {saved_count} frames from {video_file}")

# -----------------------------
# RUN EXTRACTION
# -----------------------------

print("\n--- Extracting Soccer Videos ---")
extract_from_folder(SOCCER_VIDEO_FOLDER, SOCCER_OUTPUT)

print("\n--- Extracting Non-Soccer Videos ---")
extract_from_folder(NONSOCCER_VIDEO_FOLDER, NONSOCCER_OUTPUT)

print("\n🎉 Frame extraction complete!")