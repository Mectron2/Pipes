import argparse
import os
import sys
import logging
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from google import genai
from logger import setup_logging

def highlight_force_main_image(image_path: Path) -> np.ndarray | None:
    logger = logging.getLogger(__name__)
    """
    Sends the plan image to Gemini 3.1 Flash Image Preview to highlight the FORCE MAIN line in red.
    Returns the image where the red line is drawn.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY environment variable is missing.")
        return None

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Error: Could not read image {image_path}", file=sys.stderr)
        return None

    client = genai.Client()
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    prompt = (
        "Your task is to carefully mark the main black line on this drawing, which is labeled FORCE MAIN. "
        "Highlight it in red by drawing a line that runs exactly along the FORCE MAIN curve. "
        "Ignore all other markings and objects on this drawing. "
        "Note that the pipe runs continuously from one boundary to the other. Do not break the line between them."
    )
    
    logger.info("Calling Gemini 3.1 Flash Image Preview...")
    try:
        response = client.models.generate_content(
            model='gemini-3.1-flash-image-preview',
            contents=[pil_image, prompt]
        )
    except Exception as e:
        logger.exception("Error calling Gemini API")
        return None
    
    logger.info("Received response from Gemini. Extracting image...")
    
    # Extract the image from the response parts
    for candidate in response.candidates:
        if not candidate.content:
            continue
        for part in candidate.content.parts:
            # google-genai returns image parts often as inline_data
            if hasattr(part, 'inline_data') and part.inline_data:
                image_bytes = part.inline_data.data
                np_arr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            # Check if there is an image property in the part (for some newer SDK versions)
            if hasattr(part, 'image') and part.image:
                image_bytes = part.image.image_bytes
                np_arr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
                    
    logger.error("Gemini response did not contain any valid image data.")
    return None


def highlight_force_main(image_path: Path, output_path: Path) -> bool:
    image = highlight_force_main_image(image_path)
    if image is None:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return True

def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Use Gemini to automatically highlight the FORCE MAIN line in red on a plan drawing.")
    parser.add_argument("input_image", type=Path, help="Input plan drawing image (without red line)")
    parser.add_argument("output_image", type=Path, help="Output image path to save the red-highlighted drawing")
    
    args = parser.parse_args()
    
    success = highlight_force_main(args.input_image, args.output_image)
    if success:
        logging.getLogger(__name__).info("Successfully saved highlighted image to %s", args.output_image)
        logging.getLogger(__name__).info("You can now pass this image directly to plan_to_3d.py!")
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(main())
