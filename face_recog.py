# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import os
import json
import torch
from PIL import Image as PILImage
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from transformers import BlipProcessor, BlipForConditionalGeneration
from ament_index_python.packages import get_package_share_directory

class face_recog(Node):
    def __init__(self):
        super().__init__('person_registration_node')

        self.subscription = self.create_subscription(
            Image,
            '/kinect/image_raw',
            self.image_callback,
            1
        )

        self.bridge = CvBridge()

        # YOLO model
        pkg_path = get_package_share_directory('vision')
        yolo_path = os.path.join(pkg_path, 'weights', 'yolov8n.pt')
        self.yolo_model = YOLO(yolo_path)

        # SAM model
        sam_checkpoint = os.path.join(pkg_path, 'weights', 'sam_vit_l.pth')
        self.sam = sam_model_registry["vit_l"](checkpoint=sam_checkpoint)
        self.sam_predictor = SamPredictor(self.sam)

        # BLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_dir = os.path.join(pkg_path, 'weights', 'blip2')
        self.processor = BlipProcessor.from_pretrained("salesforce/blip-image-captioning-base", cache_dir="./weights/blip")
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "salesforce/blip-image-captioning-base",
            cache_dir="./weights/blip"
        ).to(self.device)

        self.get_logger().info("face_recog service ready.")

        self.faces_dir = "./face_recog"
        os.makedirs(self.faces_dir, exist_ok=True)

        self.desc_file = os.path.join(pkg_path, 'descriptions.json')
        if not os.path.exists(self.desc_file):
            with open(self.desc_file, "w") as f:
                json.dump({}, f)

        self.registered = False

    def image_callback(self, msg):
        if self.registered:
            return 

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.yolo_model(frame)
        persons = [r for r in results[0].boxes if int(r.cls[0]) == 0]


        if not persons:
            self.get_logger().info("No se detectaron personas")
            return

        box = persons[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        person_crop = frame[y1:y2, x1:x2]

        # Segment SAM
        self.sam_predictor.set_image(person_crop)
        bbox_np = np.array([[0, 0, person_crop.shape[1], person_crop.shape[0]]]) 
        masks, _, _ = self.sam_predictor.predict(box=bbox_np, multimask_output=False)
        mask = masks[0].astype("uint8") * 255
        person_segmented = cv2.bitwise_and(person_crop, person_crop, mask=mask)

        
        name = input("Ingrese el nombre de la persona: ")
        face_path = os.path.join(self.faces_dir, f"{name}.png")
        print(f"Guardando imagen en: {face_path}")
        cv2.imwrite(face_path, person_segmented)

        # description BLIP
        pil_image = PILImage.fromarray(cv2.cvtColor(person_segmented, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        out = self.blip_model.generate(**inputs)
        description = self.processor.decode(out[0], skip_special_tokens=True)


        with open(self.desc_file, "r") as f:
            data = json.load(f)
        data[name] = description
        with open(self.desc_file, "w") as f:
            json.dump(data, f, indent=2)

        self.get_logger().info(f"Persona registrada: {name}")
        self.get_logger().info(f"Descripción: {description}")

        self.registered = True

def main(args=None):
    rclpy.init(args=args)
    node = face_recog()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
