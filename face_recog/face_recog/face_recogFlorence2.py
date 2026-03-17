# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from faces_interfaces.srv import FaceRecog
from cv_bridge import CvBridge
import numpy as np
import cv2
import os
import json
import torch
import glob
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoModelForCausalLM

class FaceServiceNode(Node):
    def __init__(self):
        super().__init__('face_service_node')
        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # --- Parameters ---
        self.declare_parameter("image_topic", "/kinect/image_raw")

        # --- Rutas ---
        base_path = os.path.expanduser("~/face_recognizer")
        self.model_dir = os.path.join(base_path, "models")
        self.faces_dir = os.path.join(base_path, "faces_detect")
        self.desc_dir = os.path.join(base_path, "descriptions")
        self.desc_file = os.path.join(self.desc_dir, "descriptions.json")

        for p in [self.faces_dir, self.desc_dir]:
            os.makedirs(p, exist_ok=True)

        # --- Carga de Modelos ---
        self.get_logger().info("MODELS BEGINING")
        model_id = "microsoft/Florence-2-base"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=self.torch_dtype).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        self.detector = cv2.FaceDetectorYN.create(os.path.join(self.model_dir, "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.9, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(os.path.join(self.model_dir, "face_recognition_sface_2021dec.onnx"), "")

        self.current_frame = None
        self.known_embeddings = {}
        self.load_known_faces()

        self.sub = self.create_subscription(ROSImage, 
                                            self.get_parameter("image_topic").value, 
                                            self.image_callback, 5)

        self.recognition_srv = self.create_service(FaceRecog, 'detect_person', self.detect_handler)
        self.indentify_srv = self.create_service(FaceRecog, 'identify_person', self.identify_handler)
        self.describe_srv = self.create_service(FaceRecog, 'describe_person', self.describe_handler)
        self.get_logger().info(">>> SERVICE READY: Waiting for requests...")

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
        
            score = area / (dist + 1e-3)
            
            if score > best_score:
                best_score = score
                best_bbox = box
        
        return best_bbox
    
    def detect_handler(self, request, response):
        frame = self.current_frame

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

        h, w, _ = frame.shape
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)

        if faces is None and len(faces) == 0:
            response.success = False
            response.description = "No person detected"
            return response
        
        for face in faces:
            face_align = self.recognizer.alignCrop(frame, face)
            feat = self.recognizer.feature(face_align)

            for name, saved_feat in self.known_embeddings.items():
                score = self.recognizer.match(feat, saved_feat, cv2.FaceRecognizerSF_FR_COSINE)
                if score > 0.363:
                    response.name = name
                    response.success = True
                    response.description = f"Identified: {name}"
                    return response
            

            response.name = "Unknown"
            response.success = False
            response.description = "Person not recognized"

    def describe_handler(self, request, response):
        frame = self.bridge.imgmsg_to_cv2(request.image, "bgr8")

        pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        result = self.run_florence("<DETAILED_CAPTION>", pil_img)
        description = result.get("<DETAILED_CAPTION>", "")

        response.successs = True
        response.description = description

        return response
    
    """
    def identify_callback(self, request, response):
        if self.current_frame is None:
            self.get_logger().error("Sin imagen de Kinect.")
            response.success = False
            return response

        frame = self.current_frame.copy()
        h, w, _ = frame.shape
        pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # detectar a las personas
        results = self.run_florence("<CAPTION_TO_PHRASE_GROUNDING>", pil_img, "person")
        bboxes = results.get('<CAPTION_TO_PHRASE_GROUNDING>', {}).get('bboxes', [])

        if not bboxes:
            response.success = False
            response.description = "don't see any person, please look to the camera"
            return response

        # Selecciona
        target = self.select_best_target(bboxes, w, h)
        tx1, ty1, tx2, ty2 = map(int, target)

        # Zoom 65
        person_h = ty2 - ty1
        zoom_ty2 = int(ty1 + (person_h * 0.65)) 
        target_crop = frame[max(0, ty1):min(h, zoom_ty2), max(0, tx1):min(w, tx2)]
        pil_target = PILImage.fromarray(cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB))

        desc_data = self.run_florence("<DETAILED_CAPTION>", pil_target)
        raw_desc = desc_data.get("<DETAILED_CAPTION>", "")

        forbidden = ["image", "show", "shows", "picture", "desk", "monitor", "office", "background", "cluttered"]
        sentences = raw_desc.split('.')
        clean_sentences = []
        for s in sentences:
            s_low = s.lower().strip()
            if not any(word in s_low for word in forbidden) and len(s_low) > 5:
                s_clean = s.strip()
                prefixes = ["The image shows ", "The person is ", "The man is ", "A young man is ", "There is "]
                for prefix in prefixes:
                    if s_clean.startswith(prefix):
                        s_clean = s_clean[len(prefix):].strip()
                if s_clean: clean_sentences.append(s_clean)
        
        core_description = ". ".join(clean_sentences) if clean_sentences else raw_desc.replace("The image shows ", "")
        identity = None 
        
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        
        if faces is not None:
            for face in faces:
                fx, fy = face[0], face[1]
                # Verificamos si la cara está dentro del bbox de la persona elegida
                if tx1 <= fx <= tx2 and ty1 <= fy <= ty2:
                    face_align = self.recognizer.alignCrop(frame, face)
                    feat = self.recognizer.feature(face_align)
                    for name, saved_feat in self.known_embeddings.items():
                        if self.recognizer.match(feat, saved_feat, cv2.FaceRecognizerSF_FR_COSINE) > 0.363:
                            identity = name
                            break
                    break

        if identity:
            final_desc = f"{identity} is {core_description}"
            response.name = identity
            response.description = final_desc
            response.success = True
            self.get_logger().info(f"Identified: {identity}")
        else:
            self.get_logger().warn("PERSON NOT KNOWN.")
            print(f"\n--- NEW REGISTRATION ---")
            print(f"I SEE: {core_description}")
            nombre = input("What is your name: ").strip()
            
            if nombre:
                cv2.imwrite(os.path.join(self.faces_dir, f"{nombre}.png"), frame)
                
                descriptions_data = {}
                if os.path.exists(self.desc_file):
                    try:
                        with open(self.desc_file, 'r') as f:
                            descriptions_data = json.load(f)
                    except: descriptions_data = {}
                
                final_desc = f"{nombre} is {core_description}"
                descriptions_data[nombre] = final_desc
                with open(self.desc_file, 'w') as f:
                    json.dump(descriptions_data, f, indent=4)
                
                self.load_known_faces()
                response.name = nombre
                response.description = final_desc
            else:
                response.name = "Unknown"
                response.description = f"Unknown person is {core_description}"
            response.success = True

        return response
        """
    
    def run_florence(self, task, pil_img, text_input=""):
        inputs = self.processor(text=task + text_input, images=pil_img, return_tensors="pt").to(self.device, self.torch_dtype)
        with torch.no_grad():
            generated_ids = self.model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=512)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return self.processor.post_process_generation(generated_text, task=task, image_size=(pil_img.width, pil_img.height))

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