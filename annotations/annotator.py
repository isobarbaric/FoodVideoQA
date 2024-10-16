from pathlib import Path
from rich.console import Console
import json
import cv2

from llm.generation import generate
from llm.frame_diff import frames as fd

from pose.localization.draw_utils import draw_text
from pose.localization.dino import make_get_bounding_boxes, determine_iou
from pose.detection.face_plotting import determine_mouth_open
from pose.detection.pose_detector import PoseDetector
from pose.eat import determine_eating

from utils.constants import FRAME_STEP_SIZE, UTENSILS

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
LLM_DATA_DIR = DATA_DIR / "llm"
LLM_VIDEO_DIR = LLM_DATA_DIR / "videos"
LLM_FRAME_DIR = LLM_DATA_DIR / "frames"

DATA_JSON = LLM_DATA_DIR / "data.json"

DETECTION_DIR = DATA_DIR / "detection"
FACE_PLOT_INFERENCE_DIR = DETECTION_DIR / "inference" 
FACE_PLOT_OUTPUT_DIR = DETECTION_DIR / "face-plots"


LOCALIZATION_DIR = DATA_DIR / "localization"
IMAGE_OUTPUT_DIR = LOCALIZATION_DIR / "outputs"


class Annotator:
    def __init__(self):
        self.console = Console()
        self.pose_detector = PoseDetector()
    
    # llm - generate stage
    def generate_json_data(self):
        questions = [
            # "Provide a detailed description of the food you see in the image.",
            "Please analyze the image and provide a comprehensive description of each food item you observe. Include details such as color, shape, texture, presentation, and any notable ingredients or garnishes. If applicable, mention the style of cuisine and any possible serving methods.",
            f"Provide a list of cutlery/utensils that the person in the image is eating with, from this list: {UTENSILS}.",
            f"Analyze the provided image and provide a list of which utensils are in the image from this list: {UTENSILS}." ,
            "Provide a detailed list of the ingredients of the food in the image. Only include a comma-separated list of items with no additional descriptions for each item in your response.",
            "Provide an approximate estimate the weight of the food in the image in grams. It is completely okay if your estimate is off, all I care about is getting an estimate. Only provide a number and the unit in your response."
        ]
        generate.process_videos(
            video_dir=LLM_VIDEO_DIR,
            frame_dir=LLM_FRAME_DIR,
            questions=questions,
            output_file=DATA_JSON,
            frame_step_size=FRAME_STEP_SIZE
        )

    # llm - frame difference stage
    def generate_frame_diff_data(self):
        fd.generate_frame_diff(input_path=DATA_JSON,
                                output_path=DATA_JSON,
                                print_output=True)
        fd.determine_eaten(DATA_JSON, print_output=True)

    def llm_generate(self):
        self.generate_json_data()
        self.generate_frame_diff_data()

    # pose - detection & localization stage
    def pose_generate(self):
        bounding_boxes = make_get_bounding_boxes()
        for video_frames in LLM_FRAME_DIR.iterdir():
            directory = video_frames.name
            for frame in video_frames.iterdir():

                bbox_output_path = IMAGE_OUTPUT_DIR / directory / f"{frame.stem}.jpg"
                face_plot_output_path = FACE_PLOT_OUTPUT_DIR / directory / f"{frame.stem}.jpg"

                if not bbox_output_path.parent.exists():
                    bbox_output_path.parent.mkdir(parents=True, exist_ok=False)
                if not face_plot_output_path.parent.exists():
                    face_plot_output_path.parent.mkdir(parents=True, exist_ok=False)

                eating, _ = determine_eating(
                    bounding_boxes, 
                    self.pose_detector, 
                    frame, 
                    bbox_output_path, 
                    face_plot_output_path
                )

                bbox_img = cv2.imread(bbox_output_path)
                if eating:
                    draw_text(
                        image = bbox_img,
                        text = 'is eating',
                        pos = (0, 20),
                        font_scale = 0.6,
                        font_thickness = 2,
                        text_color = (23, 181, 14),
                        text_color_bg = (255, 255, 255),
                        have_bg = False
                    )
                else:
                    draw_text(
                        image = bbox_img,
                        text = 'not eating',
                        pos = (0, 20),
                        font_scale = 0.6,
                        font_thickness = 2,
                        text_color = (0, 0, 255),
                        text_color_bg = (255, 255, 255),
                        have_bg = False
                    )
                cv2.imwrite(str(bbox_output_path), bbox_img)

    def extract_info(self):
        pass
    
    def annotate(self):
        pass

if __name__ == "__main__":
    annotator = Annotator()
    annotator.llm_generate()
    annotator.pose_generate()