from .gradcam import GradCAM
from .lime import LIME
import matplotlib.pyplot as plt

def visualize_explanations(original_image, input_tensor, model, class_names=None, 
                          target_class=None, save_path=None):
    """
    Create comprehensive visualization with both GradCAM and LIME.
    
    Args:
        original_image: Original image as numpy array (H, W, C) in [0, 255]
        input_tensor: Preprocessed input tensor (1, C, H, W)
        model: VGG16 model
        class_names: List of class names
        target_class: Target class for explanation
        save_path: Path to save visualization
    """
    # GradCAM
    gradcam = GradCAM(model)
    gradcam_viz, cam, pred_class = gradcam.visualize(input_tensor, original_image, target_class)
    
    # LIME
    lime_explainer = LIME(model, class_names)
    image_normalized = original_image / 255.0  # LIME expects [0, 1]
    lime_explanation = lime_explainer.explain(image_normalized)
    lime_viz, lime_label = lime_explainer.visualize(image_normalized, lime_explanation)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image.astype(np.uint8))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # GradCAM
    axes[1].imshow(gradcam_viz)
    class_name = class_names[pred_class] if class_names else f"Class {pred_class}"
    axes[1].set_title(f'GradCAM: {class_name}')
    axes[1].axis('off')
    
    # LIME
    axes[2].imshow(lime_viz)
    class_name = class_names[lime_label] if class_names else f"Class {lime_label}"
    axes[2].set_title(f'LIME: {class_name}')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    return gradcam_viz, lime_viz, pred_class
