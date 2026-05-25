import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from google import genai

def highlight_force_main(image_path: Path, output_path: Path) -> bool:
    """
    Sends the plan image to Gemini 3.1 Flash Image Preview to highlight the FORCE MAIN line in red.
    Saves the image where the red line is drawn.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is missing.", file=sys.stderr)
        return False

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Error: Could not read image {image_path}", file=sys.stderr)
        return False

    client = genai.Client()
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    prompt = (
        "Your task is to carefully mark the main black line on this drawing, which is labeled FORCE MAIN. "
        "Highlight it in red by drawing a line that runs exactly along the FORCE MAIN curve. "
        "Ignore all other markings and objects on this drawing. "
        "Note that the pipe runs continuously from one boundary to the other. Do not break the line between them."
    )
    
    print("Calling Gemini 3.1 Flash Image Preview...")
    try:
        response = client.models.generate_content(
            model='gemini-3.1-flash-image-preview',
            contents=[pil_image, prompt]
        )
    except Exception as e:
        print(f"Error calling Gemini API: {e}", file=sys.stderr)
        return False
        
    print("Received response from Gemini. Extracting image...")
    
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
                    cv2.imwrite(str(output_path), img)
                    return True
            # Check if there is an image property in the part (for some newer SDK versions)
            if hasattr(part, 'image') and part.image:
                image_bytes = part.image.image_bytes
                np_arr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is not None:
                    cv2.imwrite(str(output_path), img)
                    return True
                    
    print("Error: Gemini response did not contain any valid image data.", file=sys.stderr)
    return False

def main() -> int:
    parser = argparse.ArgumentParser(description="Use Gemini to automatically highlight the FORCE MAIN line in red on a plan drawing.")
    parser.add_argument("input_image", type=Path, help="Input plan drawing image (without red line)")
    parser.add_argument("output_image", type=Path, help="Output image path to save the red-highlighted drawing")
    
    args = parser.parse_args()
    
    success = highlight_force_main(args.input_image, args.output_image)
    if success:
        print(f"Successfully saved highlighted image to {args.output_image}")
        print("You can now pass this image directly to plan_to_3d.py!")
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(main())
