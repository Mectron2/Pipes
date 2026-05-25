from pathlib import Path
import logging


GEMINI_PLAN_MODEL = "gemini-3.1-flash-image-preview"

logger = logging.getLogger(__name__)

GEMINI_PLAN_PROMPT = """
You are processing a civil engineering pipeline plan drawing.

Task:
Identify the main target pipe route and highlight ONLY that pipe route in solid bright red.

Requirements:
- Preserve the original image size, scale, perspective, and geometry.
- Do not move, redraw, resize, or distort any lines.
- Do not modify station labels, dimensions, annotations, symbols, grids, or text.
- Do not remove existing content.
- Do not add new geometry.
- Only recolor the existing target pipe route to bright pure red (RGB 255,0,0 or similar).
- The highlighted red pipe route must remain continuous and clearly visible for OpenCV segmentation.
- Minimize all visual changes outside the target pipe route.
- Keep the engineering drawing otherwise unchanged.

Goal:
The output image will be processed by OpenCV to detect the red pipeline route and extract coordinates.
"""

GEMINI_PROFILE_PROMPT = """
You are processing a civil engineering pipeline profile drawing.

Task:
Identify the main pipeline profile line and highlight ONLY that profile line in solid bright red.

Requirements:
- Preserve the original image size, scale, perspective, axes, grid, station labels, elevation labels, annotations, and geometry.
- Do not move, redraw, resize, crop, rotate, or distort the drawing.
- Do not modify text or engineering labels.
- Do not remove existing content.
- Do not add new geometry.
- Only recolor the existing main profile line to bright pure red (RGB 255,0,0 or similar).
- The red profile line must remain continuous and easy for OpenCV to segment.
- Minimize all visual changes outside the target profile line.

Important:
- Do NOT perform OCR.
- Do NOT calculate coordinates.
- Do NOT generate CAD or CSV data.
- Only visually mark the existing profile line in red.

Goal:
The output image will be processed downstream by OpenCV for red-line extraction and coordinate calculation.
"""

def edit_image(
    input_image: Path,
    output_image: Path,
    prompt: str,
    model: str = GEMINI_PLAN_MODEL,
) -> Path:
    from google import genai
    from PIL import Image

    output_image.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Sending Gemini image edit request: input=%s output=%s model=%s", input_image, output_image, model)
    client = genai.Client()
    with Image.open(input_image) as image:
        response = client.models.generate_content(
            model=model,
            contents=[prompt, image],
        )

    for part in response.parts:
        if part.inline_data is not None:
            edited_image = part.as_image()
            edited_image.save(output_image)
            logger.info("Saved Gemini edited image: %s", output_image)
            return output_image

    raise RuntimeError("Gemini did not return an edited image")


async def edit_image_async(
    input_image: Path,
    output_image: Path,
    prompt: str,
    model: str = GEMINI_PLAN_MODEL,
) -> Path:
    from google import genai
    from PIL import Image

    output_image.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Sending Gemini image edit request: input=%s output=%s model=%s", input_image, output_image, model)
    client = genai.Client()
    try:
        with Image.open(input_image) as image:
            response = await client.aio.models.generate_content(
                model=model,
                contents=[prompt, image],
            )
    finally:
        await client.aio.aclose()

    for part in response.parts:
        if part.inline_data is not None:
            edited_image = part.as_image()
            edited_image.save(output_image)
            logger.info("Saved Gemini edited image: %s", output_image)
            return output_image

    raise RuntimeError("Gemini did not return an edited image")


def edit_plan_image(
    input_image: Path,
    output_image: Path,
    model: str = GEMINI_PLAN_MODEL,
) -> Path:
    return edit_image(input_image, output_image, GEMINI_PLAN_PROMPT, model)


async def edit_plan_image_async(
    input_image: Path,
    output_image: Path,
    model: str = GEMINI_PLAN_MODEL,
) -> Path:
    return await edit_image_async(input_image, output_image, GEMINI_PLAN_PROMPT, model)


def edit_profile_image(
    input_image: Path,
    output_image: Path,
    model: str = GEMINI_PLAN_MODEL,
) -> Path:
    return edit_image(input_image, output_image, GEMINI_PROFILE_PROMPT, model)


async def edit_profile_image_async(
    input_image: Path,
    output_image: Path,
    model: str = GEMINI_PLAN_MODEL,
) -> Path:
    return await edit_image_async(input_image, output_image, GEMINI_PROFILE_PROMPT, model)
