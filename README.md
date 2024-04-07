# LoGONE - An instance detection network for the automatic retouching of branded media 
LoGONE is an instance detection network for the automatic retouching of branded media, namely still images and videos. When LoGONE detects a brand (logo/reference/etc.), to the best of its ability, it will attempt to crop that brand from the still, generate a suitable replacement using deep learning techniques, and superimpose that replacement back on the original still. To do this, LoGONE has two fundamental components: 

## Instance Detection 
The instance detection network of LoGONE is its bread and butter. Trained off the [LogoDet-3k](https://www.kaggle.com/datasets/lyly99/logodet3k/data?select=LogoDet-3K) dataset, the instance detection network draws bounding boxes around all identified branded logos in a frame before returning these back to the user. The network structure summary is coming soon. 

## Replacement with Stable Diffusion 
The second stage of the LoGONE pipeline is to take the visual content contained in output bounding boxes, and sending this visual representation to a diffusion model for "retouching." By retouching, we seek to take the logo content and perturb it enough so that there is a clear and distinct change in the branding (i.e. no longer copywritable), while simultaneously remaining visually consistent in the image frame. This is challenging for a number of reasons, with the primary being the fact that ground truth bounding boxes for logos may not be aligned with how the logos themselves appear in the frame. An example of this phenomenon is shown below: 

![LogoDet-3k dataset overview](media/LogoDetOverview.png) 
The above image is an overview of the LogoDet-3k dataset, and can be found in the original paper, [cited here](https://arxiv.org/abs/2008.05359). 

## Getting Started: 
Instructions to come