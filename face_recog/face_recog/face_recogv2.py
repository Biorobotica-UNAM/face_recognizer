# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2
import os
import json
import torch
import numpy as np
from PIL import Image as PILImage
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from ament_index_python.packages import get_package_share_directory
import insightface

EMBEDDING_FILE = "./faces_detect/embeddings.npy"

class FaceRecogNode(Node):
    def __init__(self):
        super().__init__('face_recog_buffalo_node')

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(ROSImage, '/camera/rgb/image_raw', self.image_callback, 1)
        self.debug_pub = self.create_publisher(ROSImage, "/face/debug_image", 10)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Carpetas
        self.faces_dir = "./faces_detect"
        os.makedirs(self.faces_dir, exist_ok=True)

        self.desc_file = "./faces_detect/descriptions.json"
        if not os.path.exists(self.desc_file):
            with open(self.desc_file, "w") as f:
                json.dump({}, f)

        # Qwen2-VL
        qwen_path = "/home/angel/face_recognizer/weights/qwen2vl"
        self.qwen_processor = Qwen2VLProcessor.from_pretrained(qwen_path, local_files_only=True)
        self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            qwen_path, device_map="auto", load_in_4bit=True, local_files_only=True
        )

        # Buffalo1
        self.model_buffalo = insightface.app.FaceAnalysis(name='buffalo_l')
        self.model_buffalo.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))

        # Cargar embeddings guardados
        self.known_embeddings = {}  # {name: embedding}
        if os.path.exists(EMBEDDING_FILE):
            npz = np.load(EMBEDDING_FILE, allow_pickle=True)
            self.known_embeddings = {k: npz[k].astype(np.float32) for k in npz.files}

        self.registered = False
        self.get_logger().info("Face recognition node with Buffalo1 READY.")

    def generate_description(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)
        prompt = (
            "Describe the person in ONE short natural sentence. "
            "Focus on physical appearance, clothing, and any distinctive features. "
            "Say if the person is adult, young or child. "
            "Say if the person is male or female."
        )
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.qwen_processor(text=[text], images=[pil_image], return_tensors="pt").to(self.qwen_model.device)
        with torch.no_grad():
            outputs = self.qwen_model.generate(**inputs, max_new_tokens=120, do_sample=True, temperature=0.7, top_p=0.9)
        result = self.qwen_processor.decode(outputs[0], skip_special_tokens=True)
        result = result.replace("\n", " ").strip()
        if "assistant" in result.lower():
            result = result.split("assistant")[-1]
        result = result.replace(":", "").strip()
        if len(result) > 1:
            result = result[0].upper() + result[1:]
        return result

    def find_known_person(self, embedding, threshold=0.9):
        embedding = embedding / np.linalg.norm(embedding)
        for name, known_emb in self.known_embeddings.items():
            known_emb = known_emb / np.linalg.norm(known_emb)
            dist = np.linalg.norm(embedding - known_emb)
            if dist < threshold:
                return name
        return None

    def save_embeddings(self):
        np.savez_compressed(EMBEDDING_FILE, **self.known_embeddings)

    def image_callback(self, msg):
        if self.registered:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        faces = self.model_buffalo.get(frame)

        if not faces:
            self.get_logger().info("No face detected")
            return

        face = faces[0]
        x1, y1, x2, y2 = face.bbox.astype(int)
        face_crop = frame[y1:y2, x1:x2]

        # Comparar con embeddings conocidos
        name_known = self.find_known_person(face.embedding)
        if name_known:
            name = name_known
            self.get_logger().info(f"Recognized known person: {name}")
        else:
            name = input("What is your name? ").strip()

        # Guardar embedding
        self.known_embeddings[name] = face.embedding.astype(np.float32)
        self.save_embeddings()

        # Guardar imagen
        face_path = os.path.join(self.faces_dir, f"{name}.png")
        cv2.imwrite(face_path, face_crop)
        print(f"save image in: {face_path}")

        # Descripción
        description = self.generate_description(face_crop)
        with open(self.desc_file, "r") as f:
            data = json.load(f)
        data[name] = description
        with open(self.desc_file, "w") as f:
            json.dump(data, f, indent=2)

        # Dibujar bbox
        debug_img = frame.copy()
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        msg_debug = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
        msg_debug.header.stamp = self.get_clock().now().to_msg()
        msg_debug.header.frame_id = "camera_rgb"
        self.debug_pub.publish(msg_debug)

        self.get_logger().info(f"Person processed: {name}")
        self.get_logger().info(f"Description: {description}")

        self.registered = True  # <-- solo una vez

def main(args=None):
    rclpy.init(args=args)
    node = FaceRecogNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
