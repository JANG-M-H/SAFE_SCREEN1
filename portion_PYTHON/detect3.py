#Final1.py -> detect3.py
import whisper
from datetime import timedelta
from moviepy.editor import VideoFileClip, concatenate_audioclips
import subprocess

import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import torch

import pathlib # window에서 실행하기 위한 구문(아래 3줄)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

video_path = os.path.join("processing", "Yolotest.mp4") # 처음 넣어주는 원본 영상
output_path = 'output_video.mp4' # 완성된(모자이크, 선택 구문 무음 처리) 영상 저장 경로
next_path = os.path.join("runs", "detect", "exp", "Yolotest.mp4") # YOLO 객체 인식을 활용한 모자이크 처리된 '전체 무음' 영상

@smart_inference_mode()

# 모자이크 처리 함수(아래 10 줄)
def mosaic(src, ratio):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    mosaic_img = cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    return mosaic_img

def mosaic_area(src, x, y, width, height, ratio):
    roi = src[y:y + height, x:x + width]
    mosaic_roi = mosaic(roi, ratio)
    src[y:y + height, x:x + width] = mosaic_roi
    return src

# 선택한 단어를 무음처리 하며,,, YOLO 모자이크 처리된 전체 무음 영상에, 선택된 단어만 무음 처리한 오디오를 덮어쓰는 함수(아래 약 25줄)
def overlay_processed_audio(video_path, next_path, output_path, segments):
    # 비디오 파일 불러오기
    video = VideoFileClip(video_path) # 원본 영상 가져오기
    next_video = VideoFileClip(next_path) # YOLO 모자이크 처리된 영상 가져오기

    # video_path의 오디오 추출
    audio = video.audio # 원본 영상의 오디오 추출

    # 오디오 처리
    clips = []
    previous_end = 0

    for start, end in sorted(segments): # 원본 영상 오디오에서, 해당 단어만을 무음처리 한 오디오
        if start > previous_end:
            clips.append(audio.subclip(previous_end, start))
        clips.append(audio.subclip(start, end).volumex(0))
        previous_end = end

    if previous_end < audio.duration:
        clips.append(audio.subclip(previous_end))

    processed_audio = concatenate_audioclips(clips)

    # 오디오를 next_path의 비디오에 덮어쓰기
    next_video_with_audio = next_video.set_audio(processed_audio) # 알맞게 처리된 오디오를, 전체 무음 처리(모자이크 처리된 영상에 덮어쓰기)

    # 덮어쓴 비디오를 새 파일로 저장
    next_video_with_audio.write_videofile(output_path, codec='libx264', audio_codec='aac')

def sec_to_srt_format(seconds): # whisper 처리 시에 생성되는 '00sec' 단위를 '00H:00M:00S' 로 변경하는 함수(아래 10줄)
    # 시간을 HH:MM:SS 형식으로 변환
    time_str = str(timedelta(seconds=seconds))
    # 초 단위의 문자열을 생성
    time_str_sec = time_str[:7]
    # 밀리초를 ,로 구분된 형식으로 변환
    milliseconds = int((seconds - int(seconds)) * 1000)
    # 밀리초가 세 자리로 포맷팅되어 있으므로, 밀리초 부분을 .으로 변경
    milliseconds_str = f"{milliseconds:03d}"
    # 밀리초를 ,로 구분하여 반환
    return f"{time_str_sec},{milliseconds_str}"

# 특정 문자열 찾기 및 마스킹
def find_target(text, targets): # 해당 단어가, 무음 처리 하고 싶은, 즉 지정해 놓은 단어인가?
    return any(target in text for target in targets)

def mask_text(text, targets): # 무음 처리하고 싶은 단어를, *처리
    for target in targets:
        if target in text:
            text = text.replace(target, "*" * len(target))
    return text

# *******************************************************************************************************
# *********************** 다른 파이썬 파일에서, 내부적으로 해당 파일('detect1.py')을 실행시키는 경우 run함수에 지정된 default값들이 적용되고,
# *********************** 해당 파일을 바로 실행시키는 경우에는 parse_opt함수에 지정된 default 값들이 적용됨.
# *********************** 물론 명령어 줄에, 다른 가중치 값들을 직접 명시해주는 경우( ex) --weights qwer.pt ), 위에 것들은 무시되고, 명시해놓은 값이 적용됨
# *******************************************************************************************************

def run( #다른 파이썬 파일에서 내부적으로 이 파일을 사용하는 경우 default 값
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'processing/Yolotest.mp4',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    #save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    save_dir = ROOT / 'runs' / 'detect' / 'exp'
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)

    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            #annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        #annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            #im0 = annotator.result()
            if view_img:
                # Apply mosaic to the detected bounding boxes
                for *xyxy, _, cls in reversed(det):
                    c = int(cls)  # integer class
                    if names[c] == "pistol":  # Check if the detected object is a car (you can customize this based on your class labels)
                        x1, y1, x2, y2 = map(int, xyxy)
                        im0 = mosaic_area(im0, x1, y1, x2 - x1, y2 - y1, ratio=0.05)

                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    for *xyxy, _, cls in reversed(det):
                        c = int(cls)  # integer class
                        if names[
                            c] == "pistol":  # Check if the detected object is a car (you can customize this based on your class labels)
                            x1, y1, x2, y2 = map(int, xyxy)
                            im0 = mosaic_area(im0, x1, y1, x2 - x1, y2 - y1,
                                              ratio=0.05)  # Apply mosaic effect to the car region
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    # Apply mosaic to the detected bounding boxes
                    for *xyxy, _, cls in reversed(det):
                        c = int(cls)  # integer class
                        if names[c] == "pistol":  # Check if the detected object is a car (you can customize this based on your class labels)
                            x1, y1, x2, y2 = map(int, xyxy)
                            im0 = mosaic_area(im0, x1, y1, x2 - x1, y2 - y1,ratio=0.05)  # Apply mosaic effect to the car region
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")


    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # up

def parse_opt(): # 사용자가 명령줄에서 명시적으로 옵션을 지정하는 경우, 부분적으로 사용. 옵션을 지정하지 않으면 아래가 default값이 됨
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'zHate.pt', help='model path or triton URL')
    #parser.add_argument('--weights', nargs='+', type=str, default=[ 'zPistol.pt','zKnife.pt', 'zHandcuffs.pt', 'zCigarette.pt' ], help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'processing/Yolotest.mp4', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

print("YOLO 과정 완료")
#*****************************************여기까지 YOLO 처리*************************************************
# ********************************************************************************************************
#*****************************************여기부터 Whisper처리***********************************************

print("Whisper 모델 로드 중...")

# 모델 로드
model = whisper.load_model("medium")

# 단어 수준의 타임스탬프 추출을 위해 word_timestamps=True 설정하여 transcribe 호출
transcript = model.transcribe(video_path, word_timestamps=True)
target_words_file = "bad_word.txt"  # 타겟 단어들이 저장된 파일 경로

with open(target_words_file, 'r', encoding='utf-8') as file:
    target_strings = file.read().splitlines()

srt_entries = []
current_entry = {"start": None, "end": None, "text": ""}
segments_to_remove = [] # 무음 처리 하고 싶은 단어가 위치한 시간대를 저장하기 위한 리스트

for segment in transcript['segments']:
    for word in segment['words']:
        start_time, end_time, text = word['start'], word['end'], word['word']

        if current_entry["start"] is None:
            current_entry["start"] = start_time

        if find_target(text, target_strings):
            text = mask_text(text, target_strings) # 해당 단어 마스킹
            segments_to_remove.append((start_time, end_time))

        current_entry["end"] = end_time
        current_entry["text"] += text + " "

        if "." in text:
            srt_entries.append(current_entry)
            current_entry = {"start": None, "end": None, "text": ""}

if current_entry["text"]:
    srt_entries.append(current_entry)

# SRT 형식으로 결과 저장
with open("output.srt", "w", encoding="utf-8") as srt_file:
    for i, entry in enumerate(srt_entries, start=1):
        start = sec_to_srt_format(entry["start"])
        end = sec_to_srt_format(entry["end"])
        text = entry["text"].strip()
        srt_file.write(f"{i}\n{start} --> {end}\n{text}\n\n")

overlay_processed_audio(video_path, next_path, output_path, segments_to_remove) # 모자이크 된 영상에 제작한 음성 덮어쓰기

# 완성본 영상에 자막 달아주기 / output_path로 부터  자막 넣어줄 영상 불러오고, output.srt 파일을 뜯어서 자막 만들고, 둘을 짬뽕해서 output.mp4라는 이름으로 저장(아래 두 줄)
ffmpeg_command = f"ffmpeg -y -i {output_path} -vf \"subtitles=output.srt:force_style='OutlineColour=&H80000000,BorderStyle=4,BackColour=#FFC0504D,Outline=0,Shadow=15,MarginV=25,Fontname=Arial,Fontsize=8,Alignment=2'\" result/output.mp4"
subprocess.call(ffmpeg_command, shell=True)

print("Whisper 과정 완료")