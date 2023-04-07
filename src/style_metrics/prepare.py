from utils import * 

DATA_DIRS = ["clips", "ravdess_output"]

for DATA_DIR in DATA_DIRS:
# Running script
    split_frame_all_video_in_folder(DATA_DIR)
    generate_landmark_lips_folder(DATA_DIR)

