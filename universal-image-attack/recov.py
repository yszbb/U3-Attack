from PIL import Image
from utils import patch_initialization, mask_generation, image_grid, select_location, set_seed, recover_image
import torchvision.transforms as T

totensor = T.ToTensor()
topil = T.ToPILImage()
url_output = "image/"

for i in range(8):
    i += 1
    image_open = Image.open(url_output + "{}/0.jpg".format(i)).convert('RGB').resize((512, 512))
    init_image = Image.open(url_output + "{}/init.png".format(i)).convert('RGB').resize((512, 512))
    mask_image = Image.open(url_output + "{}/mask.png".format(i)).convert('RGB').resize((512, 512))
    img_re = recover_image(image_open, init_image, mask_image)
    img_re.save(url_output + "{}/1.png".format(i))
