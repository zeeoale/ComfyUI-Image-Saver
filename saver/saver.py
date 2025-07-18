from typing import Any, cast
from PIL.PngImagePlugin import PngInfo
from PIL.Image import Image

import json
import piexif
import piexif.helper

def save_image(image: Image, filepath: str, extension: str, quality_jpeg_or_webp: int, lossless_webp: bool, optimize_png: bool, a111_params: str, prompt: dict[str, Any] | None, extra_pnginfo: dict[str, Any] | None, embed_workflow: bool) -> None:
    if extension == 'png':
        metadata = PngInfo()
        if a111_params:
            metadata.add_text("parameters", a111_params)

        if embed_workflow:
            if extra_pnginfo is not None:
                for k, v in extra_pnginfo.items():
                    metadata.add_text(k, json.dumps(v, separators=(',', ':')))
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt, separators=(',', ':')))

        image.save(filepath, pnginfo=metadata, optimize=optimize_png)
    else: # webp & jpeg
        image.save(filepath, optimize=True, quality=quality_jpeg_or_webp, lossless=lossless_webp)

        # Native example adding workflow to exif:
        # https://github.com/comfyanonymous/ComfyUI/blob/095610717000bffd477a7e72988d1fb2299afacb/comfy_extras/nodes_images.py#L113
        pnginfo_json = {}
        prompt_json = {}
        if embed_workflow:
            if extra_pnginfo is not None:
                pnginfo_json = {piexif.ImageIFD.Make - i: f"{k}:{json.dumps(v, separators=(',', ':'))}" for i, (k, v) in enumerate(extra_pnginfo.items())}
            if prompt is not None:
                prompt_json = {piexif.ImageIFD.Model: f"prompt:{json.dumps(prompt, separators=(',', ':'))}"}

        def get_exif_bytes() -> bytes:
            exif_dict = ({
                "0th": pnginfo_json | prompt_json
                } if pnginfo_json or prompt_json else {}) | ({
                "Exif": {
                    piexif.ExifIFD.UserComment: cast(bytes, piexif.helper.UserComment.dump(a111_params, encoding="unicode"))
                },
            } if a111_params else {})
            return cast(bytes, piexif.dump(exif_dict))

        exif_bytes = get_exif_bytes()

        # JPEG format limits the EXIF bytes to a maximum of 65535 bytes
        if extension == "jpg" or extension == "jpeg":
            MAX_EXIF_SIZE = 65535
            if len(exif_bytes) > MAX_EXIF_SIZE and embed_workflow:
                print("ComfyUI-Image-Saver: Error: Workflow is too large, removing client request prompt.")
                prompt_json = {}
                exif_bytes = get_exif_bytes()
                if len(exif_bytes) > MAX_EXIF_SIZE:
                    print("ComfyUI-Image-Saver: Error: Workflow is still too large, cannot embed workflow!")
                    pnginfo_json = {}
                    exif_bytes = get_exif_bytes()
            if len(exif_bytes) > MAX_EXIF_SIZE:
                print("ComfyUI-Image-Saver: Error: Metadata exceeds maximum size for JPEG. Cannot save metadata.")
                return

        piexif.insert(exif_bytes, filepath)
