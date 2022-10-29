import image_slicer
import os

directory = 'MapsToSlice'
i = 0
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        try:
            tiles = image_slicer.slice(f, 20, save=False)
            tiles = tiles[2:-2]
            image_slicer.save_tiles(tiles, directory='Tiles', prefix='slice' + str(i), format='png')
            i += 1
        except:
            print("error")


