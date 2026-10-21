"""
Script to generate Meta Segment Anything masks.

Adapted from:
https://github.com/facebookresearch/segment-anything-2
https://github.com/facebookresearch/segment-anything

Author: Shrinivas Kulkarni

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""
import torch
import numpy as np
import cv2
import sys
import os

# SAM2 imports
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

# SAM1 imports
try:
    from segment_anything import (
        sam_model_registry,
        SamAutomaticMaskGenerator as SamAutomaticMaskGenerator_SAM1,
        SamPredictor,
    )

    SAM1_AVAILABLE = True
except ImportError:
    SAM1_AVAILABLE = False

# --- Utility Functions ---


def packBoolArray(filepath, arr):
    packed_data = bytearray()
    num_rows = len(arr)
    num_cols = len(arr[0])
    packed_data.extend(
        [num_rows >> 24, (num_rows >> 16) & 255, (num_rows >> 8) & 255, num_rows & 255]
    )
    packed_data.extend(
        [num_cols >> 24, (num_cols >> 16) & 255, (num_cols >> 8) & 255, num_cols & 255]
    )
    current_byte = 0
    bit_position = 0
    for row in arr:
        for boolean_value in row:
            if boolean_value:
                current_byte |= 1 << bit_position
            bit_position += 1
            if bit_position == 8:
                packed_data.append(current_byte)
                current_byte = 0
                bit_position = 0
    if bit_position > 0:
        packed_data.append(current_byte)
    with open(filepath, "wb") as f:
        f.write(packed_data)
    return packed_data


def saveMask(filepath, maskArr, formatBinary):
    if formatBinary:
        packBoolArray(filepath, maskArr)
    else:
        with open(filepath, "w") as f:
            for row in maskArr:
                f.write("".join(str(int(val)) for val in row) + "\n")


def saveMasks(masks, saveFileNoExt, formatBinary):
    for i, mask in enumerate(masks):
        filepath = saveFileNoExt + str(i) + ".seg"
        arr = [[val for val in row] for row in mask]
        saveMask(filepath, arr, formatBinary)


# --- Strategy Pattern Implementation ---


class SegmentationStrategy:
    def get_model_type_from_filename(self, model_filename):
        raise NotImplementedError

    def load_model(self, checkPtFilePath, modelType):
        raise NotImplementedError

    def segment_auto(self, sam, cvImage, saveFileNoExt, formatBinary, **kwargs):
        raise NotImplementedError

    def segment_box(
        self, sam, cvImage, maskType, boxCos, saveFileNoExt, formatBinary, **kwargs
    ):
        raise NotImplementedError

    def segment_sel(
        self,
        sam,
        cvImage,
        maskType,
        selFile,
        boxCos,
        saveFileNoExt,
        formatBinary,
        **kwargs,
    ):
        raise NotImplementedError

    def run_test(self, sam):
        raise NotImplementedError

    def cleanup(self):
        pass


class SAM1Strategy(SegmentationStrategy):
    MODEL_TYPE_LOOKUP = {
        "sam_vit_h_4b8939": "vit_h",
        "sam_vit_l_0b3195": "vit_l",
        "sam_vit_b_01ec64": "vit_b",
    }

    def get_model_type_from_filename(self, model_filename):
        filename_stem = os.path.splitext(model_filename)[0]
        model_type = self.MODEL_TYPE_LOOKUP.get(filename_stem)
        if model_type:
            print(f"Auto-detected SAM1 model type: {model_type}")
            return model_type
        else:
            print(
                f"Error: Could not auto-detect model type from SAM1 filename: {model_filename}"
            )
            print(
                f"Please use one of the following file names: {list(self.MODEL_TYPE_LOOKUP.keys())}"
            )
            return None

    def load_model(self, checkPtFilePath, modelType):
        if not SAM1_AVAILABLE:
            print("Error: SAM1 is not installed.")
            print(
                "Install it from https://github.com/facebookresearch/segment-anything (pip install -e .)"
            )
            return None
        try:
            sam = sam_model_registry[modelType](checkpoint=checkPtFilePath)
            print("SAM1 Model loaded successfully!")
            return sam
        except Exception as e:
            print(f"Error loading SAM1 model: {e}")
            return None

    def segment_auto(self, sam, cvImage, saveFileNoExt, formatBinary, **kwargs):
        mask_generator = SamAutomaticMaskGenerator_SAM1(sam)
        masks = mask_generator.generate(cvImage)
        masks = [mask["segmentation"] for mask in masks]
        saveMasks(masks, saveFileNoExt, formatBinary)

    def segment_box(
        self, sam, cvImage, maskType, boxCos, saveFileNoExt, formatBinary, **kwargs
    ):
        predictor = SamPredictor(sam)
        predictor.set_image(cvImage)
        input_box = np.array(boxCos)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=(maskType == "Multiple"),
        )
        saveMasks(masks, saveFileNoExt, formatBinary)

    def segment_sel(
        self,
        sam,
        cvImage,
        maskType,
        selFile,
        boxCos,
        saveFileNoExt,
        formatBinary,
        **kwargs,
    ):
        pts = []
        with open(selFile, "r") as f:
            lines = f.readlines()
            for line in lines:
                cos = line.split(" ")
                pts.append([int(cos[0]), int(cos[1])])
        predictor = SamPredictor(sam)
        predictor.set_image(cvImage)
        input_point = np.array(pts)
        input_label = np.array([1] * len(input_point))
        input_box = np.array(boxCos) if boxCos else None
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=(maskType == "Multiple"),
        )
        saveMasks(masks, saveFileNoExt, formatBinary)

    def run_test(self, sam):
        npArr = np.zeros((50, 50), np.uint8)
        cvImage = cv2.cvtColor(npArr, cv2.COLOR_GRAY2BGR)
        predictor = SamPredictor(sam)
        predictor.set_image(cvImage)
        input_box = np.array([10, 10, 20, 20])
        predictor.predict(
            point_coords=None, point_labels=None, box=input_box, multimask_output=False
        )


class SAM2Strategy(SegmentationStrategy):
    MODEL_TYPE_LOOKUP = {
        "sam2_hiera_large": "sam2_hiera_large",
        "sam2_hiera_base_plus": "sam2_hiera_base_plus",
        "sam2_hiera_small": "sam2_hiera_small",
        "sam2_hiera_tiny": "sam2_hiera_tiny",
        "sam2.1_hiera_large": "sam2_hiera_large",
        "sam2.1_hiera_base_plus": "sam2_hiera_base_plus",
        "sam2.1_hiera_small": "sam2_hiera_small",
        "sam2.1_hiera_tiny": "sam2_hiera_tiny",
    }

    def __init__(self):
        self._temp_pth_path = None

    def get_model_type_from_filename(self, model_filename):
        filename_stem = os.path.splitext(model_filename)[0]
        model_type = self.MODEL_TYPE_LOOKUP.get(filename_stem)
        if model_type:
            print(f"Auto-detected SAM2 model type: {model_type}")
            return model_type
        else:
            print(
                f"Error: Could not auto-detect model type from SAM2 filename: {model_filename}"
            )
            print(
                f"Please use one of the following file names (or their .safetensors/.pt equivalents): {list(self.MODEL_TYPE_LOOKUP.keys())}"
            )
            return None

    def _convert_safetensors_to_pth(self, safetensors_path, pth_path):
        try:
            from safetensors.torch import load_file

            state_dict = load_file(safetensors_path)
            checkpoint = {"model": state_dict}
            torch.save(checkpoint, pth_path)
            return True
        except Exception as e:
            print(f"Error converting safetensors to pth: {e}")
            return False

    def load_model(self, checkPtFilePath, modelType):
        if not SAM2_AVAILABLE:
            print("Error: SAM2 is not installed.")
            print(
                "Install it from https://github.com/facebookresearch/sam2 (pip install -e .)"
            )
            return None
        model_configs = {
            "sam2_hiera_tiny": "sam2_hiera_t.yaml",
            "sam2_hiera_small": "sam2_hiera_s.yaml",
            "sam2_hiera_base_plus": "sam2_hiera_b+.yaml",
            "sam2_hiera_large": "sam2_hiera_l.yaml",
        }
        config_file = model_configs.get(modelType, "sam2_hiera_l.yaml")
        actual_checkpoint_path = checkPtFilePath
        if checkPtFilePath.endswith(".safetensors"):
            print("Converting safetensors to pth format...")
            self._temp_pth_path = checkPtFilePath.replace(".safetensors", "_temp.pth")
            if self._convert_safetensors_to_pth(checkPtFilePath, self._temp_pth_path):
                actual_checkpoint_path = self._temp_pth_path
                print(f"Converted to: {self._temp_pth_path}")
            else:
                print("Failed to convert safetensors file")
                return None
        try:
            sam = build_sam2(config_file, actual_checkpoint_path)
            print("SAM2 Model loaded successfully!")
            return sam
        except Exception as e:
            print(f"Error loading SAM2 model: {e}")
            self.cleanup()
            return None

    def segment_auto(self, sam, cvImage, saveFileNoExt, formatBinary, **kwargs):
        points_per_side = 32
        if kwargs.get("segRes") == "Low":
            points_per_side = 16
        elif kwargs.get("segRes") == "High":
            points_per_side = 64
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            crop_n_layers=kwargs.get("cropNLayers", 0),
            min_mask_region_area=kwargs.get("minMaskArea", 0),
        )
        masks = mask_generator.generate(cvImage)
        masks = [mask["segmentation"] for mask in masks]
        saveMasks(masks, saveFileNoExt, formatBinary)

    def segment_box(
        self, sam, cvImage, maskType, boxCos, saveFileNoExt, formatBinary, **kwargs
    ):
        predictor = SAM2ImagePredictor(sam)
        predictor.set_image(cvImage)
        input_box = np.array(boxCos)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=(maskType == "Multiple"),
        )
        saveMasks(masks, saveFileNoExt, formatBinary)

    def segment_sel(
        self,
        sam,
        cvImage,
        maskType,
        selFile,
        boxCos,
        saveFileNoExt,
        formatBinary,
        **kwargs,
    ):
        pts = []
        with open(selFile, "r") as f:
            lines = f.readlines()
            for line in lines:
                cos = line.split(" ")
                pts.append([int(cos[0]), int(cos[1])])
        predictor = SAM2ImagePredictor(sam)
        predictor.set_image(cvImage)
        input_point = np.array(pts)
        input_label = np.array([1] * len(input_point))
        input_box = np.array(boxCos) if boxCos else None
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=(maskType == "Multiple"),
        )
        saveMasks(masks, saveFileNoExt, formatBinary)

    def run_test(self, sam):
        npArr = np.zeros((50, 50), np.uint8)
        cvImage = cv2.cvtColor(npArr, cv2.COLOR_GRAY2BGR)
        predictor = SAM2ImagePredictor(sam)
        predictor.set_image(cvImage)
        input_box = np.array([10, 10, 20, 20])
        predictor.predict(
            point_coords=None, point_labels=None, box=input_box, multimask_output=False
        )

    def cleanup(self):
        if self._temp_pth_path and os.path.exists(self._temp_pth_path):
            os.remove(self._temp_pth_path)
            print(f"Removed temporary file: {self._temp_pth_path}")


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python seganybridge.py <model_type|auto> <checkpoint_path> [options]"
        )
        return

    # Named options are appended by the plug-in after the positional arguments.
    options = {}
    args = []
    for arg in sys.argv[1:]:
        if arg.startswith("--prompt="):
            options["prompt"] = arg[len("--prompt=") :]
        elif arg.startswith("--"):
            print(f"Ignoring unknown option: {arg}")
        else:
            args.append(arg)

    modelType = args[0]
    checkPtFilePath = args[1]
    model_filename = os.path.basename(checkPtFilePath)

    if model_filename.lower().startswith("sam_"):
        strategy = SAM1Strategy()
    elif model_filename.lower().startswith("sam2"):
        strategy = SAM2Strategy()
    else:
        print(
            f"Error: Could not determine model family from filename: {model_filename}"
        )
        print("Filename must start with 'sam_' for SAM1 or 'sam2' for SAM2.")
        return

    if modelType.lower() == "auto":
        modelType = strategy.get_model_type_from_filename(model_filename)
        if not modelType:
            return

    if not os.path.exists(checkPtFilePath):
        print(f"Error: Checkpoint file not found: {checkPtFilePath}")
        return

    sam = strategy.load_model(checkPtFilePath, modelType)
    if sam is None:
        return

    if torch.cuda.is_available():
        try:
            sam.to(device="cuda")
            print("Model moved to CUDA")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"Cannot move the model to the GPU ({e}). Continuing on the CPU.")

    if len(args) == 2:
        strategy.run_test(sam)
        print("Success!!")
        strategy.cleanup()
        return

    ipFile = args[2]
    segType = args[3]
    maskType = args[4]
    saveFileNoExt = args[5]
    formatBinary = args[6] == "True" if len(args) > 6 else True
    textPrompt = options.get("prompt", "")

    cvImage = cv2.imread(ipFile)
    cvImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)

    try:
        if segType == "Auto":
            auto_kwargs = {"maskType": maskType, "textPrompt": textPrompt}
            if isinstance(strategy, SAM2Strategy):
                if len(args) > 7:
                    auto_kwargs["segRes"] = args[7]
                if len(args) > 8:
                    auto_kwargs["cropNLayers"] = int(args[8])
                if len(args) > 9:
                    auto_kwargs["minMaskArea"] = int(args[9])
            strategy.segment_auto(
                sam, cvImage, saveFileNoExt, formatBinary, **auto_kwargs
            )
        elif segType in {"Selection", "Box-Selection"}:
            selFile = args[7]
            boxCos = (
                [float(val.strip()) for val in args[8].split(",")]
                if len(args) > 8
                else None
            )
            strategy.segment_sel(
                sam,
                cvImage,
                maskType,
                selFile,
                boxCos,
                saveFileNoExt,
                formatBinary,
                textPrompt=textPrompt,
            )
        elif segType == "Box":
            boxCos = [float(val.strip()) for val in args[8].split(",")]
            strategy.segment_box(
                sam,
                cvImage,
                maskType,
                boxCos,
                saveFileNoExt,
                formatBinary,
                textPrompt=textPrompt,
            )
        else:
            print(f"Unknown segmentation type: {segType}")
    finally:
        print("Done!")
        strategy.cleanup()


if __name__ == "__main__":
    main()
