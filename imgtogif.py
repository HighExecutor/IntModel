import imageio
import os
images = []
for i in range(0, 1440, 10):
    images.append(imageio.imread("C:\\wspace\\projects\\IntModel\\exps_scenario\\scenario_gif\\img_{}.png".format(i)))
imageio.mimsave('C:\wspace\projects\IntModel\exps_scenario\scenario.gif', images, duration=0.0000001)