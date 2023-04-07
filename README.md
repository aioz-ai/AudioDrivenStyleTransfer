# git_style_transfer

## Dependencies 
First establish the environment and install the dependencies
```bash
conda create --prefix env/ python=3.8 -y
conda activate env/
pip install -r requirements.txt

```

Then download files from this [link](https://drive.google.com/drive/folders/16-lA4tAWkn86HkbHDgY451CWSSLizsmK?usp=sharing) and put: 
- `Obama2.zip` and `APC_epoch_160.model` in `src/face_generator/data` and extract `Obama2.zip` there.
- `GPEN-BFR-512_trace.pt`, `RealESRGAN_x2plus_trace.pt`, and `RetinaFace-R50_trace.pt` in `src/face_res/models`.
- `wiki.zip` in `src/face_res` and extract it there.
- `00000189-checkpoint.pth.tar` in `src/face_reenactment/config`.
- `shape_predictor_68_face_landmarks.dat` in `src/style_metrics`.
- `RAVDESS.zip` in `.` and then extract it there.

## Quick start 
Simply run 
```bash 
bash main.sh -i <image path> -a <audio path> -o <output path>
```
The model only accepts audio with extension `.wav` or `.mp3`, and the image must be square.

For example with the given `inputs` folder, you run:

```bash 
bash main.sh -i inputs/image.jpg -a inputs/sample.wav -o ./output
```

### TODO 
- [ ] train (coming soon) 

