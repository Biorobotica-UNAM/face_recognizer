# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from faces_interfaces.srv import FaceRecog
import cv2
from cv_bridge import CvBridge
from PIL import Image as PILImage
import numpy as np
import torch
import glob
import re
from transformers import AutoProcessor, AutoModelForCausalLM

class FaceServiceNode(Node):
    def __init__(self):
        super().__init__('face_service_node')
        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # --- Parameters ---
        self.declare_parameter("image_topic", "/kinect/image_raw") #/head_rgbd_sensor/rgb/image_rect_color && /kinect/image_raw

        # --- Rutas ---
        base_path = os.path.expanduser("~/vision")
        self.model_dir = os.path.join(base_path, "models")
        self.faces_dir = os.path.join(base_path, "faces_detected")

        # --- Model names ---
        face_detection_model = "face_detection_yunet_2023mar.onnx"
        face_recognition_model = "face_recognition_sface_2021dec.onnx"
        face_description_model = "microsoft/Florence-2-base"
        
        # --- Carga de Modelos ---
        self.get_logger().info("Loading models...")
        self.model = AutoModelForCausalLM.from_pretrained(face_description_model, trust_remote_code=True, torch_dtype=self.torch_dtype).to(self.device)
        self.processor = AutoProcessor.from_pretrained(face_description_model, trust_remote_code=True)
        
        self.detector = cv2.FaceDetectorYN.create(os.path.join(self.model_dir, face_detection_model), "", (320, 320), 0.9, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(os.path.join(self.model_dir, face_recognition_model), "")

        self.current_frame = None
        self.known_embeddings = {}
        self.load_known_faces()

        self.sub = self.create_subscription(ROSImage, 
                                            self.get_parameter("image_topic").value, 
                                            self.image_callback, 5)
        
        det_srv_name = 'detect_person'
        id_srv_name = 'identify_person'
        desc_srv_name = 'describe_person'
        reg_srv_name = 'register_person'

        self.recognition_srv = self.create_service(FaceRecog, det_srv_name, self.detect_handler)
        self.indentify_srv = self.create_service(FaceRecog, id_srv_name, self.identify_handler)
        self.describe_srv = self.create_service(FaceRecog, desc_srv_name, self.describe_handler)
        self.register_srv = self.create_service(FaceRecog, reg_srv_name, self.register_handler)

        self.get_logger().info(f"Servers ready: Waiting for requests on: {det_srv_name}, {id_srv_name}, {desc_srv_name}, {reg_srv_name}")



    def image_callback(self, msg):
        self.current_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def load_known_faces(self):
        self.known_embeddings = {}
        for img_path in glob.glob(os.path.join(self.faces_dir, "*.png")):
            name = os.path.basename(img_path).replace(".png", "")
            img = cv2.imread(img_path)
            if img is None: continue
            self.detector.setInputSize((img.shape[1], img.shape[0]))
            _, faces = self.detector.detect(img)
            if faces is not None:
                face_align = self.recognizer.alignCrop(img, faces[0])
                self.known_embeddings[name] = self.recognizer.feature(face_align)

    def select_best_target(self, bboxes, img_w, img_h):
        img_center = np.array([img_w / 2, img_h / 2])
        best_score = -float('inf')
        best_bbox = None

        for box in bboxes:
            x1, y1, x2, y2 = box
            box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            area = (x2 - x1) * (y2 - y1)
            dist = np.linalg.norm(box_center - img_center)
        
            score = area / (dist + 1)
            
            if score > best_score:
                best_score = score
                best_bbox = box
        
        return best_bbox
    
    def detect_handler(self, request, response):
        frame = self.current_frame
        
        # --- VALIDACIÓN DE FRAME ---
        if frame is None:
            self.get_logger().warn("Servicio detect_person llamado pero no hay frames de cámara.")
            response.success = False
            response.description = "No camera frame received yet"
            return response

        h, w, _ = frame.shape
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        
        if faces is not None and len(faces) > 0:
            response.success = True
            response.description = f"Detected {len(faces)} person(s)"
        else:
            response.success = False
            response.description = "No person detected"
        return response
    
    def identify_handler(self, request, response):
        frame = self.current_frame
        
        if frame is None:
            response.success = False
            response.description = "No camera frame received yet"
            return response

        h, w, _ = frame.shape
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)

        if faces is None or len(faces) == 0: # Corregido check de longitud
            response.success = False
            response.description = "No person detected"
            return response
        
        for face in faces:
            face_align = self.recognizer.alignCrop(frame, face)
            feat = self.recognizer.feature(face_align)

            for name, saved_feat in self.known_embeddings.items():
                score = self.recognizer.match(feat, saved_feat, cv2.FaceRecognizerSF_FR_COSINE)
                if score > 0.30: # Umbral de similitud (ajustar según pruebas)
                    response.name = name
                    response.success = True
                    response.description = f"Identified: {name}"
                    return response
            
        response.name = "Unknown"
        response.success = False
        response.description = "Person not recognized"
        return response

    def describe_handler(self, request, response):
        frame = self.current_frame
        if frame is None:
            response.success = False
            response.description = "No camera frame received yet"
            return response
            
        h, w, _ = frame.shape
        pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        results = self.run_florence("<CAPTION_TO_PHRASE_GROUNDING>", pil_img, "person")
        bboxes = results.get('<CAPTION_TO_PHRASE_GROUNDING>', {}).get('bboxes', [])

        if not bboxes:
            response.success = False
            response.description = "No person detected"
            return response

        target = self.select_best_target(bboxes, w, h)
        tx1, ty1, tx2, ty2 = map(int, target)

        crop = frame[max(0, ty1):min(h, ty2), max(0, tx1):min(w, tx2)]
        pil_crop = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        result = self.run_florence("<DETAILED_CAPTION>", pil_crop)
        raw_output = result.get("<DETAILED_CAPTION>", "").lower()
        self.get_logger().info(f"FLORENCE RAW: {raw_output}")
        
        # Genero
        gender = "male" if any(x in raw_output for x in ["man", "boy", "male"]) else "female" if any(x in raw_output for x in ["woman", "girl", "female"]) else "person"
        
        # Color de playera
        colors = ["black", "white", "red", "blue", "green", "yellow", "gray", "grey", "navy", "orange", "purple"]
        shirt_color = "unknown color"
        for c in colors:
            if c in raw_output:
                shirt_color = f"{c} shirt"
                break

        # Edad
        age = "adult"
        if "young" in raw_output or "boy" in raw_output or "girl" in raw_output:
            age = "young"
        elif "elderly" in raw_output or "old" in raw_output:
            age = "elderly"

        # Accesorios
        acc_found = []
        if "chain" in raw_output or "necklace" in raw_output: acc_found.append("chain")
        if "watch" in raw_output: acc_found.append("watch")
        if "glasses" in raw_output or "spectacles" in raw_output: acc_found.append("glasses")
        if "hat" in raw_output or "cap" in raw_output: acc_found.append("hat")
        if "glasses" in raw_output or "spectacles" in raw_output: acc_found.append("glasses")
        if "headphones" in raw_output or "headset" in raw_output: acc_found.append("headphones")
        if "beard" in raw_output or "mustache" in raw_output: acc_found.append("facial hair")
        
        accessories = ", ".join(acc_found) if acc_found else "no accessories"
        final_result = f"{gender}, {shirt_color}, {age}, {accessories}"
        response.success = True
        response.description = final_result
        return response

    def run_florence(self, task, pil_img, text_input=""):
        prompt = task + text_input
        
        inputs = self.processor(text=prompt, images=pil_img, return_tensors="pt").to(self.device, self.torch_dtype)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"], 
                pixel_values=inputs["pixel_values"], 
                max_new_tokens=128, # Más tokens para que no se corte la descripción
                do_sample=False,
                num_beams=3
            )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return self.processor.post_process_generation(generated_text, task=task, image_size=(pil_img.width, pil_img.height))
    
    def register_handler(self, request, response):
        if not request.name.strip():
            response.success = False
            response.description = "No name provided"
            return response

        frame = self.current_frame
        h, w, _ = frame.shape

        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)

        if faces is None or len(faces) == 0:
            response.success = False
            response.description = "No face detected"
            return response


        face = max(faces, key=lambda f: f[2] * f[3])

        face_align = self.recognizer.alignCrop(frame, face)
        feat = self.recognizer.feature(face_align)

        name = request.name.strip()

        img_path = os.path.join(self.faces_dir, f"{name}.png")
        cv2.imwrite(img_path, frame)

        self.load_known_faces()

        response.success = True
        response.name = name
        response.description = f"{name} registered successfully"

        self.get_logger().info(f"Registered new person: {name}")

        return response
  

def main(args=None):
    rclpy.init(args=args)
    node = FaceServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()