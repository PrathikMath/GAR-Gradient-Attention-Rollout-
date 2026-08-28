Gradient Attention Rollout (GAR)

    Gradient Attention Rollout (GAR) is an explainability technique designed for Transformer-
based deepfake detection models. 

    It combines attention information from Transformer layers with gradient information to 
identify the image regions that have the greatest influence on the model's prediction.

    GAR generates a heatmap that highlights suspicious or manipulated regions in an image.
Areas with higher activation indicate regions that contributed more strongly to the
model's decision. 
 
    This makes the deepfake detector more interpretable by allowing users to visually 
understand where the model detected potential manipulation.

    In a deepfake detection system, GAR can be used after the Transformer makes its 
prediction.

    The resulting heatmap can be overlaid on the original face image, providing a visual 
explanation of the prediction and helping users distinguish between genuine facial features
and potentially manipulated regions.

Key advantages:

- Provides visual explanations for Transformer predictions.
- Highlights regions that influence the model's decision.
- Helps identify localized facial manipulations.
- Improves transparency and interpretability of deepfake detection.
- Can be useful for analyzing whether the model focuses on meaningful facial regions rather
  than irrelevant features.