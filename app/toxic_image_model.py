from nudenet import NudeClassifierLite

classifier = NudeClassifierLite()

def detect_nsfw_image(image_path):
    result = classifier.classify(image_path)
    return result
