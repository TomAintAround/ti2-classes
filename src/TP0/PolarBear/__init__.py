import matplotlib.pyplot as plt
import numpy as np
import math


def showImg(img: np.ndarray) -> None:
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.axis("off")


def enhanceColor(img: np.ndarray, factor: float, channel: int) -> np.ndarray:
    imgE: np.ndarray = img.copy()
    imgCh: np.ndarray = imgE[:, :, channel]
    imgCh = imgCh.astype(np.float64) * factor
    imgCh = np.clip(imgCh, 0, 255)
    imgCh = imgCh.astype(np.uint8)
    imgE[:, :, channel] = imgCh

    return imgE


def imgMosaic(img: np.ndarray, width: float) -> np.ndarray:
    imgM: np.ndarray = img.copy()
    imgHeight, imgWidth, _ = imgM.shape
    for line in range(0, imgHeight, math.floor(width)):
        for column in range(0, imgWidth, math.floor(width)):
            yRange: int = int(min(line + width, imgHeight))
            xRange: int = int(min(column + width, imgWidth))
            imgM[line:yRange, column:xRange] = imgM[line, column]

    return imgM


def color2gray(img: np.ndarray) -> np.ndarray:
    imgHeight, imgWidth, _ = img.shape
    imgG: np.ndarray = np.zeros([imgHeight, imgWidth], "uint8")
    imgG[:, :] = (
        0.2978 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    )

    return imgG


def imagemBin(img: np.ndarray, limiar: int) -> np.ndarray:
    imgB: np.ndarray = img.copy()
    imgB[imgB[:, :] >= limiar] = 255
    imgB[imgB[:, :] < limiar] = 0

    return imgB


def main() -> None:
    plt.close("all")
    img: np.ndarray = plt.imread("src/TP0/PolarBear/polarbear.jpg")

    showImg(img)
    imgE: np.ndarray = enhanceColor(img, 10.8, 2)
    showImg(imgE)
    imgM: np.ndarray = imgMosaic(img, 30)
    showImg(imgM)
    imgG: np.ndarray = color2gray(imgE)
    showImg(imgG)
    imgB: np.ndarray = imagemBin(imgG, 85)
    showImg(imgB)
    plt.show()

    plt.imsave("src/TP0/PolarBear/polarbear.bmp", imgB, cmap="gray")


if __name__ == "__main__":
    main()
