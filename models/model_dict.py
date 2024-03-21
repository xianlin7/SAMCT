from models.segment_anything.build_sam import sam_model_registry
from models.segment_anything_samct.build_samct import samct_model_registry
from models.segment_anything_samct_autoprompt.build_samct import autosamct_model_registry

def get_model(modelname="SAM", args=None, opt=None):
    if modelname == "SAM":
        model = sam_model_registry['vit_b'](checkpoint=args.sam_ckpt)
    elif modelname == "SAMCT":
        model = samct_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    elif modelname == "AutoSAMCT":
        model = autosamct_model_registry['vit_b'](args=args, checkpoint=opt.load_path)
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model
