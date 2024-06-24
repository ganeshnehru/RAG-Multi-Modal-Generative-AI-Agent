from PIL import Image
import io
import base64


def resize_image_to_base64(image_path, max_height=750):
    """
    Resizes an image to retain its aspect ratio with a maximum height of 750 pixels,
    then encodes the resized image to base64.

    :param image_path: Path to the input image.
    :param max_height: Maximum height for the resized image.
    :return: Base64 encoded string of the resized image.
    """
    # Open the image file in binary mode
    with open(image_path, "rb") as image_file:
        with Image.open(image_file) as img:
            # Calculate the new height and width while retaining the aspect ratio
            aspect_ratio = img.width / img.height
            new_height = min(max_height, img.height)
            new_width = int(aspect_ratio * new_height)

            # Resize the image
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)

            # Save the resized image to a bytes buffer
            buffer = io.BytesIO()
            resized_img.save(buffer, format='JPEG')

            # Encode the image to base64
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return img_base64


# # Example usage
# image_base64 = resize_image_to_base64('/Users/ganesh/Downloads/1-min.jpg')
# print(len(image_base64))
