from PIL import Image, ImageEnhance

def enhanceImage(image):
    # make the image sharper
    Im = Image.open('stitchedOutputProcessed.png')
    enhancer = ImageEnhance.Sharpness(Im)
    enhanced = enhancer.enhance(5.0)
    enhanced.save('enhanced.png')

    return enhanced