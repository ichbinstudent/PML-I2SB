from typing import Literal, cast

from src.options import Options

def build_degradations(opt: Options, degradation_type: str):
    if degradation_type is None: degradation_type = opt.degradation

    if "inpaint" in degradation_type:
        from .inpainting import build_inpaint_center, build_inpaint_freeform
        # Extract mask type after "inpaint-"
        mask = degradation_type.replace("inpaint-", "")
        
        # Map user-friendly names to internal names
        mask_mapping = {
            "center": "center",
            "freeform1020": "10-20% freeform",
            "freeform2030": "20-30% freeform",
            "freeform3040": "30-40% freeform",
        }
        
        if mask not in mask_mapping:
            raise ValueError(f"Unknown inpainting mask type: {mask}. "
                           f"Available types: {list(mask_mapping.keys())}")
        
        internal_mask = mask_mapping[mask]
        
        if internal_mask == "center":
            method = build_inpaint_center(opt)
        elif "freeform" in internal_mask:
            mask_lit = cast(Literal["10-20% freeform", "20-30% freeform", "30-40% freeform"], internal_mask)
            method = build_inpaint_freeform(opt, mask_lit)
        else:
            raise NotImplementedError(f"Degradation type {degradation_type} not implemented.")
    elif "blur" in degradation_type:
        from .blur import build_blur
        kernel_type = degradation_type.split("-")[1]
        if kernel_type not in {"uni", "gauss"}:
            raise ValueError(f"Unknown blur kernel type: {kernel_type}. Available types: ['uni', 'gauss']")
        method = build_blur(opt, cast(Literal["uni", "gauss"], kernel_type))
    elif "jpeg" in degradation_type:
        from .jpeg import build_jpeg
        quality = int(degradation_type.split("-")[1])
        method = build_jpeg(quality)
    elif "superres" in degradation_type:
        from .superres import build_superres
        sr_filter = degradation_type.split("-")[1]
        method = build_superres(opt, cast(Literal["bicubic", "bilinear", "pool"], sr_filter))
    else:
        raise ValueError(f"Unknown degradation type: {degradation_type}")
    
    return method
