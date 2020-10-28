import os
from PIL import Image

path = "."
dirs = os.listdir(path)


def gif_order (data, center=True):
    gif = []
    base = 1

    #primera mitad
    i = int((len(data)-2)/2)
    while(i > base ):
        gif.append(data[i])
        #print(i)
        i -= 1


    #el del medio izq
    gif.append(data[int((len(data)-2)/2) + 1])
    #print(int((len(data)-2)/2) + 1)

    #el inicial
    if center:
        gif.append(data[0])
    #print(0)

    # el del medio der
    gif.append(data[int((len(data) - 2) / 2) + 2])
    #print(int((len(data) - 2) / 2) +2)
    #segunda mitad
    i = int((len(data)-2)/2) + 3
    while (i < len(data)):
        gif.append(data[i])
        #print(i)
        i += 1
    #print("---------")

    invertedgif = gif[::-1]
    invertedgif = invertedgif[1:]

    gif = gif[1:] + invertedgif
    #print(gif)
    #for image in gif:
    #    image.show()
    #gsdfgsfgf
    return gif

# This would print all the files and directories
for file in dirs:
    if ".jpg" in file or ".png" in file:
        rowImages = []
        im = Image.open("./" + file)
        width, height = im.size
        im = im.convert('RGB')

        #CROP (left, top, right, bottom)

        pointleft = 3
        pointtop = 3
        i = 0
        while (pointtop < height):
            while (pointleft < width):
                im1 = im.crop((pointleft, pointtop, 128+pointleft, 128+pointtop))
                rowImages.append(im1.quantize())
                #im1.show()
                pointleft+= 128+4
            # Ya tengo todas las imagenes podria hacer el gif aca
            rowImages = gif_order(rowImages,center=False)
            name = file[:-4] + "_" + str(i) + '.gif'
            rowImages[0].save(name, save_all=True,format='GIF', append_images=rowImages[1:], optimize=True, duration=100, loop=0)
            pointtop += 128 + 4
            pointleft = 3
            rowImages = []
            i+=1
        #im2 = im.crop((width / 2, 0, width, height))
        # im2.show()

        #im1.save("./2" + file[:-4] + ".png")
        #im2.save("./" + file[:-4] + ".png")

    # Deleted
    #os.remove("data/" + file)