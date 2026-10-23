import os
import sys
from pathlib import Path

from pip._vendor import truststore as t
t.extract_from_ssl()

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# Load the existing backend environment variables.
BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

# Reuse the existing Figure 38 image; do not reprocess the PDF.
IMAGE_PATH = (
    BACKEND_DIR
    / "data"
    / "diagrams"
    / "integration-test"
    / "fig38_1.png"
)

sys.path.insert(0, str(BACKEND_DIR))

from ingest.embed_and_store_docling import _describe_image

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Figure 38 image not found: {IMAGE_PATH}")

# Relevant text transcribed from page 91 of the ABB manual.
context = """
Section 7.6.4.1: Conversion of measured insulation resistance values.

To compare measured insulation resistance values, the values are
stated at 40°C. The measured value is converted to the corresponding
40°C value using the diagram. Use the diagram for temperatures fairly
near 40°C because large deviations can cause errors.

The diagram's numbered labels are:
1. Coefficient Kt for Insulation Resistance.
2. Winding temperature, degrees Celsius (°C).

R = Insulation resistance value at a specific temperature.
R40 = Equivalent insulation resistance at 40°C.
R40 = k x R.

Printed example:
R = 30 MΩ measured at 20°C.
k = 0.25.
R40 = 0.25 x 30 MΩ = 7.5 MΩ.
"""

def main():
    with Image.open(IMAGE_PATH) as image:
        image.load()
        client = OpenAI()
        description = _describe_image(
            client,
            image,
            context=context,
        )

    print("\n--- Figure 38 VLM Description ---")
    print(description or "No description was returned.")

if __name__ == "__main__":
    main()