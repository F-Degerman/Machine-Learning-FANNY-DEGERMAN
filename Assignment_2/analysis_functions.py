import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask


# -----------------------------
# Model and input preparation
# -----------------------------

# Load pretrained ResNet18, matching preprocessing, and ImageNet class names.
def setup_model(device="cpu"):
    weights = ResNet18_Weights.DEFAULT

    model = models.resnet18(weights=weights).to(device)
    model.eval()

    preprocess = weights.transforms()
    class_names = weights.meta["categories"]

    return model, weights, preprocess, class_names


# -----------------------------
# Prediction
# -----------------------------

# Run regular inference and return logits, probabilities, and top-k predictions.
def predict_image(model, input_tensor, class_names, k=5):
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)

    logits = logits.detach().cpu()
    probs = probs.detach().cpu()
    top_probs, top_idxs = probs.topk(k, dim=1)

    topk = []

    for rank, (prob, idx) in enumerate(zip(top_probs[0], top_idxs[0]), start=1):
        class_id = idx.item()

        topk.append({
            "rank": rank,
            "class_id": class_id,
            "label": class_names[class_id],
            "prob": prob.item(),
        })

    return {
        "logits": logits,
        "probs": probs,
        "topk": topk,
    }

# -----------------------------
# Class activation maps
# -----------------------------

# Generate a CAM result for the target class.
# The result includes the activation map, heatmap, and overlay image.
def generate_cam(model, img, input_tensor, target_class):
    model.eval()
    cam_extractor = SmoothGradCAMpp(model)

    try:
        with torch.enable_grad():
            scores = model(input_tensor)

        activation_map = cam_extractor(target_class, scores)[0]
        heatmap = to_pil_image(activation_map.squeeze(0).detach().cpu(), mode="F")
        overlay = overlay_mask(img, heatmap, alpha=0.5)

    finally:
        cam_extractor.remove_hooks()

    return {
        "target_class": target_class,
        "activation_map": activation_map.detach().cpu(),
        "heatmap": heatmap,
        "overlay": overlay,
        "scores": scores.detach().cpu(),
    }


# -----------------------------
# Layer activations
# -----------------------------

# Collect activations from selected model layers.
def get_layer_activations(model, input_tensor, layer_names):
    activations = {}
    hooks = []
    modules = dict(model.named_modules())

    for layer_name in layer_names:
        layer = modules[layer_name]

        def hook(module, inputs, output, name=layer_name):
            activations[name] = output.detach().cpu()

        hooks.append(layer.register_forward_hook(hook))

    model.eval()

    with torch.no_grad():
        model(input_tensor)

    for hook in hooks:
        hook.remove()

    return activations

# Reduce feature maps from each layer into 2D images.
def summarize_feature_maps(activations, reduction="mean"):
    summaries = {}

    for layer_name, fmap in activations.items():
        fmap = fmap[0]

        if reduction == "mean":
            summary = fmap.mean(dim=0)
        elif reduction == "max":
            summary = fmap.max(dim=0).values
        else:
            raise ValueError("reduction must be 'mean' or 'max'")

        summaries[layer_name] = summary

    return summaries


# -----------------------------
# Visualization
# -----------------------------

# Show the input image and print top-k predictions.
def show_topk(img, pred_result):
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Input image")
    plt.show()

    print("Top predictions:")
    for row in pred_result["topk"]:
        print(f"{row['rank']}. {row['label']}: {row['prob']:.4f}")

# Show the CAM overlay image.
def show_cam_result(cam_result, class_names=None):
    target_class = cam_result["target_class"]

    if class_names is not None:
        title = f"CAM for: {class_names[target_class]}"
    else:
        title = f"CAM for class id {target_class}"

    plt.figure(figsize=(5, 5))
    plt.imshow(cam_result["overlay"])
    plt.axis("off")
    plt.title(title)
    plt.show()

# Show summarized feature maps for multiple layers.
def show_layer_summaries(layer_summaries, cmap="viridis"):
    n = len(layer_summaries)
    plt.figure(figsize=(4 * n, 4))

    for i, (layer_name, summary) in enumerate(layer_summaries.items(), start=1):
        plt.subplot(1, n, i)
        plt.imshow(summary, cmap=cmap)
        plt.title(layer_name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Print top-k logits with class names.
def print_topk_logits(pred_result, class_names, k=10):
    logits = pred_result["logits"]
    top_vals, top_idxs = logits.topk(k, dim=1)

    print("Top logits:")
    for i in range(k):
        class_id = top_idxs[0, i].item()
        logit = top_vals[0, i].item()
        label = class_names[class_id]
        print(f"{i + 1}. {label}: logit={logit:.4f}")


# -----------------------------
# Full analysis pipelines
# -----------------------------

# Run the full analysis pipeline for one image.
# This function is used by analyze_many when several images are analyzed.
def analyze_image(
    image_path,
    model,
    preprocess,
    class_names,
    target_class,
    layer_names=("layer1", "layer2", "layer3", "layer4"),
    k=5,
    fmap_reduction="mean",
    device="cpu",
):
    # Load the image and convert it to a model-ready tensor. 
    img = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    pred_result = predict_image(model, input_tensor, class_names, k=k)
    cam_result = generate_cam(model, img, input_tensor, target_class=target_class)
    activations = get_layer_activations(model, input_tensor, layer_names)
    layer_summaries = summarize_feature_maps(activations, reduction=fmap_reduction)

    return {
        "image_path": image_path,
        "img": img,
        "input_tensor": input_tensor.detach().cpu(),
        "pred": pred_result,
        "cam": cam_result,
        "activations": activations,
        "layer_summaries": layer_summaries,
    }

# Run analyze_image for several images.
def analyze_many(image_paths, **kwargs):
    analyses = []

    for image_path in image_paths:
        analyses.append(analyze_image(image_path=image_path, **kwargs))

    return analyses
