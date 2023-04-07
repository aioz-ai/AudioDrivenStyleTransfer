
[![Generic badge](https://img.shields.io/badge/python-3.6%20|%203.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Generic badge](https://img.shields.io/badge/pytorch-1.10.0-red.svg)](https://www.tensorflow.org/install)
[![Generic badge](https://img.shields.io/badge/onnxruntime-1.10.0-silver.svg)](https://www.tensorflow.org/install)
[![Generic badge](https://img.shields.io/badge/CUDA-10.1-green.svg)](https://developer.nvidia.com/cuda-10.1-download-archive-update2)

# Face Restoration

| **Author(s)**   | Truong Le (Stephen Lee)                                                                                     |
|:----------------|:------------------------------------------------------------------------------------------------------------|
| **Reviewer(s)** | Chien Duong                                                                                                 |
| **Start Date**  | Dec , 2021                                                                                                  |
| **Topic(s)**    | AI / Deep Tech / General Techniques                                                                         |
| **Status**      | **In Progress**                                                                                             |
| **Report link** | [ppt](https://docs.google.com/presentation/d/1oY2izAQcoUIrAWOLdVD0PCVfygGN0RkGKT-Qyuiqn-w/edit?usp=sharing) |


- System include three modules:
    - Super-Resolution: ESRGAN - Enhanced Super-Resolution Generative Adversarial Networks.
    - Face-Detection: Detection face and landmark
    - FaceGan: Face Restoration - GPEN

# Table Of Content

- [Pipe line](#pipe-line)
- [Setup](#setup)
    - [Install package](#install-package)
    - [Download model](#download-model)
- [Run demo](#run-demo)
    - [Run with python](#run-with-python)
        - [Run with image input](#run-with-image-input)
        - [Run with video input](#run-with-video-input)
    - [Run with Postman](#run-with-postman)
- [Processing Time and Resource](#processing-time-and-resource)
    - [Processing time](#processing-time)
    - [Resource](#resource)
- [Results](#results)
- [TODO](#todo)
- [Reference](#reference)
- [License](#license)

# Pipe line

![pipeLine](wiki/pipeline.png)

# Setup
## Install package
Follow commandline

```
pip install -r requirements.txt
```

## Download model

- Download model at [aiar-smb](smb://192.168.1.60/aiar-drive/AIAR2021/NFT/FaceRestoration/models) and put it at folder `models`
- Check information in `config.py`

# Run demo

- The system includes 3 modules: **super-resolution**, **face detection**, **face restoration**. For each module, we provide some format for run.
- So before running the demo, you must be check [API_CFG](/config.py#L160), and change the format perfect for you ([list format](/config.py#L31))
- Description for [list format](/config.py#L31)
    - Super-resolution:
        + [real_ESRGANx4](/config.py#L107): ESRGAN super-resolution x4 with torch.load format (original).
        + [real_ESRGANx4_trace](/config.py#L116): ESRGAN super-resolution x4 with torch.jit.trace format (inference without architecture script).
        + [real_ESRGANx4_ONNX](/config.py#L124): ESRGAN super-resolution x4 with onnxruntime format.
        + [real_ESRGANx2](/config.py#L131): ESRGAN super-resolution x2 with torch.load format (original).
        + [real_ESRGANx2_trace](/config.py#L139): ESRGAN super-resolution x2 with torch.jit.load format (inference without architecture script).
        + [real_ESRGANx2_ONNX](/config.py#L147): ESRGAN super-resolution x2 with onnxruntime format.
    - Face detection:
        + [retina_faceDet_trace](/config.py#L41): Retina Face detection with format torch.jit.trace
    - Face restoration:
        + [GPEN_256](/config.py#L87): Face resoration, face size 256, with torch.load format (original)
        + [GPEN_512](/config.py#L55): Face resoration, face size 512, with torch.load format (original)
        + [GPEN_512_Trace](/config.py#L67): Face resoration, face size 512, with torch.jit.trace format (inference without architecture script)
        + [GPEN_512_ONNX](/config.py#L79): Face resoration, face size 512, with onnxruntime format.

> With torch.jit.trace and onnxruntime format, they will use fewer resources, <br/>
> with these formats we can use C++ to read the model and the implementation is also simpler.

## Run with python

### Run with image input

```
python3 demo.py wiki/er.jpg save
```

### Run with video input

```
python3 demo.py wiki/vid_test.mp4 save sound
```

## Run with Postman

- Run server

```
python3 server.py
```

Server will be ready on `localhost:8088` (change port at [config.py](/config.py#L173))

- Use postman send request
    - Route: `FaceRestorationImage` for input is image and `FaceRestorationVideo` for input is video
    - Format input:

| Key              | Value               | Description                                           | 
|:-----------------|:--------------------|:------------------------------------------------------|
| file             | Image or video file | file need to restoration                              |
| super_resolution | True / False        | Set True if  using super-resolution module            |
| keep_sound       | True / False        | Set True if you want to get video response have sound |

- Response: Image or Video file

- Example

![postman-interface](wiki/postman_vid.png)

# Processing Time and Resource

## Processing time

| **Option \ Modules**                            | Detection | Super-resolution | Face restoration | Full system       |
|:------------------------------------------------|:----------|:-----------------|:-----------------|:------------------|
| original                                        | -         | -                | -                | 1.2s ~ 0.8 fps    |
| Fix size input                                  | -         | -                | -                | 0.2s ~ 5 fps      |
| Fix size input - <br/> Ignore super-rescolution | -         | -                | -                | **0.11s ~ 9 fps** |

## Resource

>***UPDATING***

# Results

|           Input            |   Output with super-resolution module    |     Output without super-resolution module      |
|:--------------------------:|:----------------------------------------:|:-----------------------------------------------:|
|     ![er](wiki/er.jpg)     |     ![er_srTrue](wiki/er_srTrue.jpg)     |       ![er_srFalse](wiki/er_srFalse.jpg)        |
| ![test](wiki/test_02.jpeg) | ![test_srTrue](wiki/test_02_srTrue.jpeg) |   ![test_srFalse](wiki/test_02_srFalse.jpeg)    |
| [video](wiki/vid_test.mp4) |                    -                     | [video output](wiki/vid_test_srFalse_sound.mp4) |


>***NOTE***: If the input image has a large aspect ratio of the face, we recommend that you skip the super-resolution module (set flag `super_resolution` is False)


# TODO

- [x] Compress video output
- [x] Convert model to another format: trace, onnxruntime
- [ ] Convert model to half precision
- [ ] Wrapper module detection with format ONNX
- [ ] Convert post-process, pre-process to C++
- [ ] Change detection module
- [ ] Measure processing time and resource for each module

# Reference

- [GPEN](https://github.com/yangxy/GPEN)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [Torch.Jit](https://pytorch.org/docs/stable/jit.html)
- [ONNX Runtime](https://onnxruntime.ai/)

# License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
  <img alt="License Creative Commons " style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
</a><br />
This repo is shared with terms of 
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
  Creative Commons Attribution 4.0 International (CC BY 4.0)
</a> @ <a rel="author" href="https://ai.aioz.io/"> AIOZ Pte Ltd </a>
