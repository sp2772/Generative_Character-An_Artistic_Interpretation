import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm

# --------------------------------------------
# SETUP
# --------------------------------------------
# Make sure pytorch-yolo-v3 submodule path is available
# sys.path.append("./pytorch-yolo-v3")

# from darknet import Darknet
# from util import write_results, load_classes
# from preprocess import letterbox_image

import importlib.util
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pytorch-yolo-v3"))

# Path to pytorch-yolo-v3 directory
YOLO_DIR = os.path.join(os.path.dirname(__file__), "pytorch-yolo-v3")

def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

darknet = import_from_path("darknet", os.path.join(YOLO_DIR, "darknet.py"))
util = import_from_path("util", os.path.join(YOLO_DIR, "util.py"))
preprocess = import_from_path("preprocess", os.path.join(YOLO_DIR, "preprocess.py"))

# Now you can use:
Darknet = darknet.Darknet
write_results = util.write_results
letterbox_image = preprocess.letterbox_image

# --------------------------------------------
# DETECTOR CLASS
# --------------------------------------------
class AnimeHeadDetector:
    def __init__(self, cfgfile, weightsfile, conf_thresh=0.85, nms_thresh=0.4, inp_dim=512):
        self.CONFIDENCE_THRESHOLD = conf_thresh
        self.NMS_THRESHOLD = nms_thresh
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = Darknet(cfgfile)
        self.model.load_weights(weightsfile)
        self.model.net_info["height"] = inp_dim
        self.model.to(self.device)
        self.model.eval()

        self.inp_dim = inp_dim
        self.classes = ["Head"]

    def detect(self, img):
        orig_h, orig_w = img.shape[:2]
        img_letterboxed = letterbox_image(img, (self.inp_dim, self.inp_dim))
        img_tensor = img_letterboxed[:, :, ::-1].transpose((2, 0, 1)).copy()  # BGR→RGB, HWC→CHW
        img_tensor = torch.from_numpy(img_tensor).float().div(255.0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(img_tensor, CUDA=torch.cuda.is_available())

        output = write_results(
            prediction,
            self.CONFIDENCE_THRESHOLD,
            len(self.classes),
            nms=True,
            nms_conf=self.NMS_THRESHOLD
        )

        if type(output) == int:
            return []

        # Rescale boxes to original image
        output = output.cpu().numpy()
        boxes = []
        for det in output:
            x1, y1, x2, y2 = det[1:5]
            x1 = max(0, int(x1 * orig_w / self.inp_dim))
            x2 = min(orig_w, int(x2 * orig_w / self.inp_dim))
            y1 = max(0, int(y1 * orig_h / self.inp_dim))
            y2 = min(orig_h, int(y2 * orig_h / self.inp_dim))
            boxes.append((x1, y1, x2, y2))
        return boxes


# --------------------------------------------
# BATCH CROP FUNCTION
# --------------------------------------------
# (Old version commented out)
# def extract_all_faces(input_dir, output_dir, cfgfile, weightsfile):
#     os.makedirs(output_dir, exist_ok=True)
#     detector = AnimeHeadDetector(cfgfile, weightsfile)
#     img_files = [...]
#     ...

def extract_all_faces(input_dir, output_dir, cfgfile, weightsfile):
    os.makedirs(output_dir, exist_ok=True)

    no_face_dir = os.path.join(output_dir, "no_face_detected")
    os.makedirs(no_face_dir, exist_ok=True)

    read_fail_dir = os.path.join(output_dir, "read_fail")
    os.makedirs(read_fail_dir, exist_ok=True)

    failed_crops_dir = os.path.join(output_dir, "failed_crops")
    os.makedirs(failed_crops_dir, exist_ok=True)

    detector = AnimeHeadDetector(cfgfile, weightsfile)

    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not img_files:
        print("No images found in input folder.")
        return

    total = 0
    read_fail = 0
    no_detections = 0
    successful_crops = 0
    detection_but_no_successful_crop = 0

    for img_name in tqdm(img_files, desc="Processing images"):
        total += 1
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            read_fail += 1
            try:
                open(os.path.join(read_fail_dir, img_name), "wb").close()
            except Exception:
                pass
            continue

        boxes = detector.detect(img)

        # Case A: no boxes
        if not boxes:
            no_detections += 1
            cv2.imwrite(os.path.join(no_face_dir, img_name), img)
            continue

        # Case B: detections exist
        base_name = os.path.splitext(img_name)[0]
        saved_any_for_this_image = False

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            box_w = x2 - x1
            box_h = y2 - y1

            pad_w = int(0.2 * box_w / 2)
            pad_h = int(0.2 * box_h / 2)

            new_x1 = max(0, x1 - pad_w)
            new_y1 = max(0, y1 - pad_h)
            new_x2 = min(img.shape[1], x2 + pad_w)
            new_y2 = min(img.shape[0], y2 + pad_h)

            new_x1, new_y1, new_x2, new_y2 = map(int, (new_x1, new_y1, new_x2, new_y2))

            if new_x2 <= new_x1 or new_y2 <= new_y1:
                continue

            face_crop = img[new_y1:new_y2, new_x1:new_x2]
            if face_crop is None or face_crop.size == 0:
                continue

            out_name = f"{base_name}_p{i+1}.jpg" if len(boxes) > 1 else f"{base_name}.jpg"
            out_path = os.path.join(output_dir, out_name)

            if os.path.exists(out_path):
                k = 1
                base_out = os.path.splitext(out_name)[0]
                while True:
                    candidate = f"{base_out}_{k}.jpg"
                    candidate_path = os.path.join(output_dir, candidate)
                    if not os.path.exists(candidate_path):
                        out_path = candidate_path
                        break
                    k += 1

            cv2.imwrite(out_path, face_crop)
            saved_any_for_this_image = True
            successful_crops += 1

        if not saved_any_for_this_image:
            detection_but_no_successful_crop += 1
            cv2.imwrite(os.path.join(failed_crops_dir, img_name), img)

    print("===== Summary =====")
    print(f"Total input images: {total}")
    print(f"Successfully saved crops: {successful_crops}")
    print(f"Images with no detections -> saved to no_face_dir: {no_detections}")
    print(f"Images where boxes existed but no valid crop produced -> saved to failed_crops: {detection_but_no_successful_crop}")
    print(f"Images unreadable by cv2 (read_fail): {read_fail}")
    print("===================")


# --------------------------------------------
# MAIN
# --------------------------------------------
if __name__ == "__main__":
    INPUT_DIR = r"vivian_nsfw_tagged_working"
    OUTPUT_DIR = "./Vivian_nsfw_cropped_faces"
    CFG_FILE = r"AnimeHeadDetector\head.cfg"
    WEIGHTS_FILE = r"AnimeHeadDetector\head.weights"

    extract_all_faces(INPUT_DIR, OUTPUT_DIR, CFG_FILE, WEIGHTS_FILE)
    print("✅ All faces extracted and saved in:", OUTPUT_DIR)
