from typing import List, Tuple
import cv2
from ultralytics import YOLO
import numpy as np
from ultralytics.utils.plotting import Annotator, colors
import os

class WeaponImageProcessor:
    """
    A class for processing images to detect weapons using YOLOv8 models.
    The class loads a YOLOv8 model for weapon detection, processes images,
    and applies blurring to detected regions where weapons are identified.
    """

    def __init__(self) -> None:
        pass

    def load_yolo_model(self, model_path: os.path) -> YOLO:
        """
        Load the YOLOv8 model for weapon detection.

        Args:
            model_path (os.path): Path to the YOLOv8 weights file.

        Returns:
            YOLO: An instance of the YOLO model loaded with the specified weights.
        """
        self.model = YOLO(model_path)
        return self.model

    def blur_detected_weapons(self, names, frame, annotator, box, cls, blur_ratio) -> np.ndarray:
        """
        Apply a blur effect to a detected weapon region in the image.

        Args:
            names (List[str]): List of class names detected by the model.
            frame (np.ndarray): The image containing the detected weapon.
            annotator (Annotator): Instance of Annotator for drawing boxes and labels.
            box : Coordinates of the detected weapon bounding box.
            cls (int): Class index of the detected object (weapon).
            blur_ratio (int): Ratio for blurring the weapon region.

        Returns:
            np.ndarray: Image with the weapon region blurred.
        """

        annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])
        obj = frame[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
        blur_obj = cv2.blur(obj, (blur_ratio, blur_ratio))

        frame[int(box[1]): int(box[3]), int(box[0]): int(box[2])] = blur_obj

        return frame

    def process_image(self, image_path: str, blur_ratio: int = 50) -> None:
        """
        Detect weapons in an image and blur the detected weapon regions.

        Args:
            image_path (str): Path to the image file.
            blur_ratio (int): Ratio for blurring weapon regions.
        """
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Error: Could not read image {image_path}.")
            return

        model = self.load_yolo_model("best.pt")

        results = model.track(frame, persist=True, verbose=True)
        if results[0].boxes:  # Check if there are any boxes detected
            boxes = results[0].boxes.xyxy.cpu()
            clss = results[0].boxes.cls.cpu().tolist()
            scores = results[0].boxes.conf.cpu().tolist()
            names = model.names

            annotator = Annotator(frame, line_width=2, example=names)

            if boxes is not None:
                for box, cls, score in zip(boxes, clss, scores):
                    if score >= 0.001:
                        frame = self.blur_detected_weapons(names, frame, annotator, box, cls, blur_ratio)

        cv2.imshow("Weapon Detection Output", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_file = 'image.png'
    yolov8_model_path = 'yolov8s.pt'
    processor = WeaponImageProcessor()
    processor.process_image(image_path=image_file,blur_ratio=30)


