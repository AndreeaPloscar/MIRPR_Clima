import os

from PIL import Image

# Giving The Original image Directory
# Specified

directory = 'Warm'
i = 0
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        try:
            Original_Image = Image.open(f)
            # Rotate Image By 180 Degree
            rotated_image1 = Original_Image.rotate(180)

            # This is Alternative Syntax To Rotate
            # The Image
            rotated_image2 = Original_Image.transpose(Image.ROTATE_90)

            # This Will Rotate Image By 60 Degree
            rotated_image3 = Original_Image.rotate(60)

            rotated_image1.save("rotated/" + directory + "/" + str(i) + ".png", 'PNG')
            i += 1
            rotated_image2.save("rotated/" + directory + "/" + str(i) + ".png", 'PNG')
            i += 1
            rotated_image3.save("rotated/" + directory + "/" + str(i) + ".png", 'PNG')
            # rotated_image2.show()
            # rotated_image3.show()
            i += 1
        except:
            print("error")
