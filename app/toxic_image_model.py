from nudenet import NudeClassifierLite

# Initialize once to avoid reloading the model for every request
classifier = NudeClassifierLite()

def detect_nsfw_image(image_path: str) -> float:
    """
    Detects NSFW score from an image file using NudeClassifierLite.
    
    Args:
        image_path (str): Path to the image file.

    Returns:
        float: NSFW score (between 0 and 1).
    """
    result = classifier.classify(image_path)
    score = result.get(image_path, {}).get("unsafe", 0.0)
    return round(score, 4)
