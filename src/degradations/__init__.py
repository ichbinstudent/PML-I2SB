from src.options import Options

def build_degradations(opt: Options, degradation_type: str):
    if degradation_type is None: degradation_type = opt.degradation

    if "inpaint" in degradation_type:
        from .inpainting import build_inpaint_center, build_inpaint_freeform
        mask = degradation_type.split("-")[1]
        assert mask in ["center", "10-20% freeform", "20-30% freeform", "30-40% freeform"]
        if mask == "center":
            method = build_inpaint_center(opt)
        elif "freeform" in mask:
            method = build_inpaint_freeform(opt, mask)
        else:
            raise NotImplementedError(f"Degradation type {degradation_type} not implemented.")
    elif "blur" in degradation_type:
        from .blur import build_blur
        kernel_type = degradation_type.split("-")[1]
        method = build_blur(opt, kernel_type)
    elif "jpeg" in degradation_type:
        from .jpeg import build_jpeg
        quality = int(degradation_type.split("-")[1])
        method = build_jpeg(quality)
    elif "superres" in degradation_type:
        from .superres import build_superres
        sr_filter = degradation_type.split("-")[1]
        method = build_superres(opt, sr_filter, image_size=opt.image_size)
