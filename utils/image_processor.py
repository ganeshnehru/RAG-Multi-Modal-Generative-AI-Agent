from PIL import Image, ImageOps
import base64
import io

class ImageProcessor:

    def resize_encode_image(image_path, desired_size=256):
        """
        Resize an image while maintaining the aspect ratio and padding to the desired size.

        Args:
            image_path (str): The file path to the input image.
            desired_size (int): The desired size for the image.

        Returns:
            PIL.Image: The resized image.
        """
        image = Image.open(image_path)
        image.thumbnail((desired_size, desired_size), Image.Resampling.LANCZOS)
        image = ImageOps.pad(image, (desired_size, desired_size), color="white")

        # Encode the image in base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        b64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return b64_string