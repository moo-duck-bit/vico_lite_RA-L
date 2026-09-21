"""Check that the segmentation model used for non-oracle perception loads.

Run from the repository root:

    python detection_pipeline/test_install.py [image]

Without an argument it runs on one of the bundled demo frames.
"""

import sys

from mmdet.apis import inference_detector, init_detector

CONFIG = "detection_pipeline/config.py"
CHECKPOINT = "detection_pipeline/epoch_1.pth"
DEFAULT_IMAGE = "demo/demo_images/img_0.jpg"


def main() -> None:
    image = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    model = init_detector(CONFIG, CHECKPOINT, device="cpu")
    print(inference_detector(model, image))


if __name__ == "__main__":
    main()
