# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import numpy as np
import cv2
import os
import json
import torch
from PIL import Image as PILImage
from ultralytics import YOLO
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
#from transformers import BlipProcessor, BlipForConditionalGeneration
from ament_index_python.packages import get_package_share_directory

class face_recog(Node):
    def __init__(self):
        super().__init__('person_registration_node')

        self.bridge = CvBridge()
        self.last_image = None
        self.subscription = self.create_subscription(ROSImage, '/kinect/image_raw', self.image_callback, 1)
        #self.subscription = self.create_subscription(ROSImage, '/camera/rgb/image_raw', self.image_callback, 1)
        self.debug_pub = self.create_publisher(ROSImage, "/face/debug_image", 10)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.waiting_for_person = True

        # YOLO model
        pkg_path = get_package_share_directory('face_recog')
        yolo_path = os.path.join(pkg_path, 'weights', 'yolo11m-seg.pt')
        self.yolo_model = YOLO(yolo_path)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


        # BLIP model
        """blip_dir = os.path.join(pkg_path, 'weights', 'blip2')
        self.processor = BlipProcessor.from_pretrained("salesforce/blip-image-captioning-base", cache_dir="./weights/blip")
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "salesforce/blip-image-captioning-base",
            cache_dir="./weights/blip"
        ).to(self.device)"""
        
        qwen_path = "/home/angel/face_recognizer/weights/qwen2vl"
        self.qwen_processor =  Qwen2VLProcessor.from_pretrained(qwen_path, local_files_only=True)
        self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(qwen_path, device_map="auto", load_in_4bit=True, local_files_only=True)
        #self.qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        #self.qwen_model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", device_map="auto", load_in_4bit=True)
        self.get_logger().info("face_recog service ready.")

        self.faces_dir = "./faces_detect"
        os.makedirs(self.faces_dir, exist_ok=True)

        self.desc_file = os.path.join(pkg_path, 'descriptions.json')
        if not os.path.exists(self.desc_file):
            with open(self.desc_file, "w") as f:
                json.dump({}, f)

        self.registered = False

    def cb_image(self, msg):
        self.last_image = msg

    def generate_description(self, image_bgr):

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)

        prompt = (
            "Describe the person in ONE short natural sentence. "
            "Focus on physical appearance, clothing, and any distinctive features. "
            "describe if the person is adult, young, child"
            "describe if the person is male or female"
        )

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.qwen_processor(text=[text], images=[pil_image], return_tensors="pt").to(self.qwen_model.device)

        with torch.no_grad():
            outputs = self.qwen_model.generate(**inputs, max_new_tokens=120, do_sample=True, temperature=0.7, top_p=0.9)

        result = self.qwen_processor.decode(outputs[0], skip_special_tokens=True)

        #clean result
        if "assistant" in result.lower():
            result = result.split("assistant")[-1]
        result = result.replace(":", "").strip()
        result = result.replace("\n", " ").strip()
        bad_starts = [
            "the person is",
            "this person is",
            "a person is",
            "the image shows",
            "i see",
        ]

        r = result.lower()
        for b in bad_starts:
            if r.startswith(b):
                result = result[len(b):].strip()
                break

        if len(result) > 1:
            result = result[0].upper() + result[1:]
        return result

    def image_callback(self, msg):

        if self.registered:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.yolo_model(frame)
        debug_img = frame.copy()

        if results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # solo personas si quieres
                if cls_id != 0:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(
                    debug_img,
                    f"pendejo  {conf:.2f}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

        persons = [r for r in results[0].boxes if int(r.cls[0]) == 0]

        #segmentation yolo
        if not persons:
            if self.waiting_for_person:
                self.get_logger().info("not found person")
                self.waiting_for_person = False
            return

        box = persons[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        person_crop = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2 , 5)

        if len(faces) == 0:
            self.get_logger().info("no face detected, please look to the camera")
            return

        mask = results[0].masks 
        if mask is None:
            self.get_logger().info("segmentation mask not found")
            return
        
        mask = mask.data[0].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)

        mask_crop = mask[y1:y2, x1:x2]
        person_segmented = cv2.bitwise_and(person_crop, person_crop, mask=mask_crop)
    

        name = input("what is your name? ")
        face_path = os.path.join(self.faces_dir, f"{name}.png")
        print(f"save image in: {face_path}")
        cv2.imwrite(face_path, person_segmented)

        # description BLIP
        """pil_image = PILImage.fromarray(cv2.cvtColor(person_segmented, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        out = self.blip_model.generate(**inputs)
        description = self.processor.decode(out[0], skip_special_tokens=True)"""

        description = self.generate_description(person_segmented)


        with open(self.desc_file, "r") as f:
            data = json.load(f)
        data[name] = description
        with open(self.desc_file, "w") as f:
            json.dump(data, f, indent=2)

        self.get_logger().info(f"person registred: {name}")
        self.get_logger().info(f"Description: {name} is {description}")
        self.registered = True

        msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "kinect_rgb"
        self.debug_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = face_recog()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
