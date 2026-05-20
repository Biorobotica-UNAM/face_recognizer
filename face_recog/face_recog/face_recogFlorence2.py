# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage, CameraInfo
from faces_interfaces.srv import FaceRecog
import cv2
from cv_bridge import CvBridge
from PIL import Image as PILImage
import numpy as np
import torch
import glob
import json
from datetime import datetime
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig


class FaceServiceNode(Node):
    def __init__(self):
        super().__init__('face_service_node')
        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # --- Parameters ---
        self.declare_parameter("image_topic", "/kinect/image_raw")
        self.declare_parameter("info_topic", "/kinect/camera_info")

        # --- Paths ---
        base_path = os.path.expanduser("~/vision")
        self.model_dir = os.path.join(base_path, "models")
        self.faces_dir = os.path.join(base_path, "faces_detected")
        self.report_path = os.path.join(base_path, "person_report.json")
        os.makedirs(self.faces_dir, exist_ok=True)

        # --- Model names ---
        face_detection_model  = "face_detection_yunet_2023mar.onnx"
        face_recognition_model = "face_recognition_sface_2021dec.onnx"
        face_description_model = "microsoft/Florence-2-base"

        # --- Load models ---
        self.get_logger().info("Loading models...")
        model_config = AutoConfig.from_pretrained(face_description_model, trust_remote_code=True)
        if not hasattr(model_config, 'forced_bos_token_id'):
            setattr(model_config, 'forced_bos_token_id', None)

        self.model = AutoModelForCausalLM.from_pretrained(
            face_description_model,
            config=model_config,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            face_description_model, trust_remote_code=True)

        det_path = os.path.join(self.model_dir, face_detection_model)
        rec_path = os.path.join(self.model_dir, face_recognition_model)
        self.detector   = cv2.FaceDetectorYN.create(det_path, "", (320, 320), 0.9, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(rec_path, "")

        self.current_frame   = None
        self.known_embeddings = {}
        self.load_known_faces()

        self.intrinsics = None
        self.info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter("info_topic").value,
            self.info_callback, 10)
        self.sub = self.create_subscription(
            ROSImage,
            self.get_parameter("image_topic").value,
            self.image_callback, 5)
        self.debug_pub = self.create_publisher(ROSImage, "/face_recog/debug_image", 10)

        self.recognition_srv = self.create_service(FaceRecog, 'detect_person',   self.detect_handler)
        self.identify_srv    = self.create_service(FaceRecog, 'identify_person',  self.identify_handler)
        self.describe_srv    = self.create_service(FaceRecog, 'describe_person',  self.describe_handler)
        self.register_srv    = self.create_service(FaceRecog, 'register_person',  self.register_handler)

        self.get_logger().info("Servers ready: detect_person, identify_person, describe_person, register_person")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def info_callback(self, msg):
        self.intrinsics = {
            'fx': msg.k[0], 'fy': msg.k[4],
            'cx': msg.k[2], 'cy': msg.k[5]
        }
        self.get_logger().info(f"Camera Info linked: fx={self.intrinsics['fx']:.2f}")
        self.destroy_subscription(self.info_sub)

    def image_callback(self, msg):
        self.current_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def publish_debug_image(self, frame, bbox=None, label=None):
        debug_img = frame.copy()
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if label:
                cv2.putText(debug_img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))

    def load_known_faces(self):
        self.known_embeddings = {}
        for img_path in glob.glob(os.path.join(self.faces_dir, "*.png")):
            name = os.path.basename(img_path).replace(".png", "")
            img  = cv2.imread(img_path)
            if img is None:
                continue
            self.detector.setInputSize((img.shape[1], img.shape[0]))
            _, faces = self.detector.detect(img)
            if faces is not None:
                face_align = self.recognizer.alignCrop(img, faces[0])
                self.known_embeddings[name] = self.recognizer.feature(face_align)
        self.get_logger().info(f"Loaded {len(self.known_embeddings)} known faces.")

    def save_report(self, name, description):
        """Persist description to JSON report keyed by person name."""
        data = {}
        if os.path.exists(self.report_path):
            try:
                with open(self.report_path, "r") as f:
                    data = json.load(f)
            except Exception as e:
                self.get_logger().error(f"Error loading report: {e}")

        data[name] = {
            "description": description,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(self.report_path, "w") as f:
            json.dump(data, f, indent=4)

    def select_best_target(self, bboxes, img_w, img_h):
        """Return the bbox closest to image center weighted by area."""
        img_center = np.array([img_w / 2, img_h / 2])
        best_score, best_bbox = -float('inf'), None
        for box in bboxes:
            x1, y1, x2, y2 = box
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            area   = (x2 - x1) * (y2 - y1)
            dist   = np.linalg.norm(center - img_center)
            score  = area / (dist + 1)
            if score > best_score:
                best_score, best_bbox = score, box
        return best_bbox

    # ── Description parser ────────────────────────────────────────────────────

    def parse_description(self, raw: str) -> str:
        """
        Extract structured attributes from Florence free-text output.

        Strategy for clothing color: scan for a (color, garment) pair where
        the color word appears *before* the garment word and within 60 chars,
        which matches natural English ("wearing a blue shirt").  Falls back to
        proximity search if no preceding color is found.
        """
        PRENDAS = ["t-shirt", "shirt", "jersey", "jacket", "hoodie", "sweater", "polo"]
        COLORES = ["green", "red", "black", "white", "blue", "yellow",
                   "gray", "grey", "navy", "orange", "purple", "pink", "brown"]
        HAIR_COLORS = ["black", "dark", "brown", "blonde", "gray", "grey", "white", "red"]

        # Gender
        if any(x in raw for x in ["man", "boy", " male"]):
            gender = "male"
        elif any(x in raw for x in ["woman", "girl", "female"]):
            gender = "female"
        else:
            gender = "person"

        # Clothing — prefer color that precedes the garment within 60 chars
        shirt_color = "unknown clothing"
        prenda_found = "shirt"
        for p in PRENDAS:
            if p in raw:
                prenda_found = p
                break

        pos_prenda = raw.find(prenda_found)
        if pos_prenda != -1:
            best_color = None
            best_dist  = 999

            for c in COLORES:
                pos_c = raw.find(c)
                if pos_c == -1:
                    continue
                dist = pos_prenda - pos_c          # positive → color before garment
                if 0 < dist < 60:                  # color precedes garment (natural order)
                    if dist < best_dist:
                        best_dist  = dist
                        best_color = c
                        
            if best_color is None:
                # Fallback: nearest color in either direction
                for c in COLORES:
                    pos_c = raw.find(c)
                    if pos_c == -1:
                        continue
                    dist = abs(pos_prenda - pos_c)
                    if dist < best_dist:
                        best_dist  = dist
                        best_color = c

            if best_color:
                shirt_color = f"{best_color} {prenda_found}"

        # Hair
        hair_color = next((c for c in HAIR_COLORS if c in raw), "")
        if "bald" in raw:
            hair = "bald"
        else:
            style = ("messy" if any(x in raw for x in ["messy", "disheveled"])
                     else "short" if "short" in raw
                     else "long"  if "long"  in raw
                     else "")
            if style and hair_color:
                hair = f"{style} {hair_color} hair"
            elif hair_color:
                hair = f"{hair_color} hair"
            else:
                hair = "dark hair"

        # Emotion
        if any(x in raw for x in ["smile", "smiling", "happy"]):
            emotion = "happy"
        elif "sad" in raw:
            emotion = "sad"
        elif "angry" in raw:
            emotion = "angry"
        elif "surprised" in raw:
            emotion = "surprised"
        elif any(x in raw for x in ["serious", "seriously"]):
            emotion = "serious"
        else:
            emotion = "neutral"

        # Age
        if any(x in raw for x in ["young", "boy", "girl", "teen"]):
            age = "young"
        elif any(x in raw for x in ["elderly", "old man", "old woman"]):
            age = "elderly"
        else:
            age = "adult"

        # Accessories
        acc = []
        if any(x in raw for x in ["chain", "necklace"]):  acc.append("chain")
        if "watch" in raw:                                  acc.append("watch")
        if any(x in raw for x in ["glasses", "spectacles"]): acc.append("glasses")
        if any(x in raw for x in ["hat", "cap"]):           acc.append("hat")
        if any(x in raw for x in ["headphones", "headset"]): acc.append("headphones")
        if any(x in raw for x in ["beard", "mustache"]):    acc.append("facial hair")
        accessories = ", ".join(acc) if acc else "no accessories"

        return f"{gender}, {shirt_color}, {age}, {hair}, {emotion}, {accessories}"

    # ── Service handlers ──────────────────────────────────────────────────────

    def detect_handler(self, request, response):
        frame = self.current_frame
        if frame is None:
            response.success     = False
            response.description = "No camera frame received yet"
            return response

        h, w, _ = frame.shape
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)

        valid_faces = [f for f in faces if f[2] > 40] if faces is not None else []

        if valid_faces:
            best_face        = max(valid_faces, key=lambda f: f[2] * f[3])
            response.success = True
            response.description = f"Detected {len(valid_faces)} person(s)."
            response.bbox = [
                float(best_face[0]),
                float(best_face[1]),
                float(best_face[0] + best_face[2]),
                float(best_face[1] + best_face[3])
            ]
            response.z = 0.0
            self.publish_debug_image(frame, bbox=response.bbox, label="Person_Detected")
        else:
            response.success     = False
            response.description = "No person detected"
            response.bbox        = [0.0, 0.0, 0.0, 0.0]
            response.z           = 0.0

        return response

    def identify_handler(self, request, response):
        frame = self.current_frame
        if frame is None:
            response.success     = False
            response.description = "No camera frame received yet"
            return response

        h, w, _ = frame.shape
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)

        if faces is None or len(faces) == 0:
            response.success     = False
            response.description = "No person detected"
            return response

        # Process faces largest-first
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

        for face in faces:
            face_align = self.recognizer.alignCrop(frame, face)
            feat       = self.recognizer.feature(face_align)

            best_name  = "Unknown"
            max_score  = -1.0

            for name, saved_feat in self.known_embeddings.items():
                score = self.recognizer.match(
                    feat, saved_feat, cv2.FaceRecognizerSF_FR_COSINE)
                if score > max_score:
                    max_score = score
                    best_name = name

            if max_score > 0.5:
                response.success     = True
                response.name        = best_name
                response.bbox        = [
                    float(face[0]), float(face[1]),
                    float(face[0] + face[2]), float(face[1] + face[3])
                ]
                response.z           = 0.0
                response.description = f"Identified: {best_name} (score: {max_score:.2f})"
                self.publish_debug_image(
                    frame, bbox=response.bbox,
                    label=f"{best_name} ({max_score:.2f})")
                return response

        response.name        = "Unknown"
        response.success     = False
        response.description = "Person not recognized (score too low)"
        return response

    def describe_handler(self, request, response):
        frame = self.current_frame
        if frame is None:
            response.success     = False
            response.description = "No camera frame received yet"
            return response

        h, w, _ = frame.shape
        pil_img  = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        results = self.run_florence("<CAPTION_TO_PHRASE_GROUNDING>", pil_img, "person")
        bboxes  = results.get('<CAPTION_TO_PHRASE_GROUNDING>', {}).get('bboxes', [])

        if not bboxes:
            response.success     = False
            response.description = "No person detected"
            return response

        tx1, ty1, tx2, ty2 = map(int, self.select_best_target(bboxes, w, h))
        crop     = frame[max(0, ty1):min(h, ty2), max(0, tx1):min(w, tx2)]
        pil_crop = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        result     = self.run_florence("<MORE_DETAILED_CAPTION>", pil_crop)
        raw_output = result.get("<MORE_DETAILED_CAPTION>", "").lower()
        self.get_logger().info(f"FLORENCE DESCRIPTION: {raw_output}")

        final_result = self.parse_description(raw_output)

        # FIX: use request.name (passed by caller) instead of last_name_processed
        # so the report is always keyed by the correct person.
        report_name = request.name.strip() if request.name.strip() else "Unknown"
        self.save_report(report_name, final_result)

        response.success     = True
        response.description = final_result
        return response

    def register_handler(self, request, response):
        if not request.name.strip():
            response.success     = False
            response.description = "No name provided"
            return response

        frame = self.current_frame
        if frame is None:
            response.success     = False
            response.description = "No frame"
            return response

        h, w, _ = frame.shape
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)

        if faces is None or len(faces) == 0:
            response.success     = False
            response.description = "No face detected"
            return response

        face     = max(faces, key=lambda f: f[2] * f[3])
        name     = request.name.strip()
        img_path = os.path.join(self.faces_dir, f"{name}.png")
        cv2.imwrite(img_path, frame)

        # Reload immediately so identify works right after register
        self.load_known_faces()

        response.success     = True
        response.name        = name
        response.description = f"{name} registered successfully"
        self.get_logger().info(f"Registered new person: {name}")
        self.publish_debug_image(
            frame,
            bbox=[float(face[0]), float(face[1]),
                  float(face[0] + face[2]), float(face[1] + face[3])],
            label=f"{name} Registered")
        return response

    # ── Florence inference ────────────────────────────────────────────────────

    def run_florence(self, task, pil_img, text_input=""):
        prompt = task + text_input
        inputs = self.processor(
            text=prompt, images=pil_img, return_tensors="pt"
        ).to(self.device, self.torch_dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=350,
                do_sample=True,
                temperature=0.8,
                num_beams=3,
                top_p=0.9,
                early_stopping=True
            )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False)[0]
        return self.processor.post_process_generation(
            generated_text, task=task,
            image_size=(pil_img.width, pil_img.height))


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