import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import huffmancodec as huffc


def compareMPG(data, varnames, ind):
    plt.subplot(3, 2, ind + 1)
    plt.plot(data[:, ind], data[:, -1], "*")
    plt.xlabel(varnames[ind])
    plt.ylabel(varnames[-1])
    plt.title(varnames[-1] + " vs. " + varnames[ind])


def getOccurrences(valueList):
    # occurrencesNums = np.zeros(valueList.max() + 1, dtype=np.uint16)
    # for i in range(len(valueList)):
    #     occurrencesNums[valueList[i]] += 1
    # return occurrencesNums
    return np.bincount(valueList)


def showOccurrences(occurrencesList, i, varnames):
    occurrencesNums = occurrencesList[i]
    alphabet = np.arange(len(occurrencesNums))
    alphabet = alphabet[occurrencesNums > 0]
    count = occurrencesNums[occurrencesNums > 0]
    plt.figure()
    plt.bar(alphabet.astype("str"), count)
    plt.xlabel(varnames[i])
    plt.ylabel("Count")
    plt.show()


def binning(column, binningSize):
    binned = column.copy()
    binnedCounts = getOccurrences(binned)
    for i in range(0, binned.max() + 1, binningSize):
        interval = binnedCounts[i : i + binningSize]
        mostCommon = interval.argmax() + i
        binned[(binned >= i) & (binned < i + binningSize)] = mostCommon
    return binned


def entropyAndProbability(occurrencesNums):
    probability = occurrencesNums / occurrencesNums.sum()
    probability = probability[probability > 0]
    entropy = -np.sum(probability * np.log2(probability))
    return entropy, probability


def huffmanEnconde(column, probability):
    codec = huffc.HuffmanCodec.from_data(column)
    _, lengths = codec.get_code_len()
    lengths = np.array(lengths)

    # Media ponderada
    meanLength = np.sum(probability * lengths)
    variance = np.sum(probability * (lengths - meanLength) ** 2)

    return meanLength, variance


def corrcoef(data, ind):
    return np.corrcoef(data[:, ind], data[:, -1])[1, 0]


def mutualInfo(data, entropyList, ind):
    column = data[:, ind]
    mpg = data[:, -1]
    probMatrix = np.zeros((column.max() + 1, mpg.max() + 1), dtype=np.float64)
    for i in range(len(column)):
        probMatrix[column[i], mpg[i]] += 1

    # column = data[:, ind]
    # mpg = data[:, -1]
    # xVals = np.unique(column)
    # yVals = np.unique(mpg)
    # probMatrix, _, _ = np.histogram2d(column, mpg, bins=[xVals, yVals])
    # # ^ isto e mais eficiente, mas por alguma razao menos preciso

    # dataFrame = pd.DataFrame(data)
    # probMatrix = pd.crosstab(dataFrame.iloc[:, ind], dataFrame.iloc[:, -1])
    # probMatrix = probMatrix.to_numpy(dtype=np.float64)
    # # Sinceramente nao percebo como isto funciona

    matrixEntropy, _ = entropyAndProbability(probMatrix)
    # I(X, Y)  = H(X)             + H(Y)            -
    mutualInfo = entropyList[ind] + entropyList[-1] - matrixEntropy

    return mutualInfo


def estimateMPG(data, biggestMiIndex, smallestMiIndex):
    mpgOriginal = data[:, -1]
    mpgEst1 = np.zeros_like(mpgOriginal, dtype=np.float64)
    mpgEst1 += -5.5241
    mpgEst2 = mpgEst1.copy()
    mpgEst3 = mpgEst1.copy()
    mpgEstList = [mpgEst1, mpgEst2, mpgEst3]

    coefficients = [-0.146, -0.4909, 0.0026, -0.0045, 0.6725, -0.0059]
    for i in range(len(coefficients)):
        mpgEst1 += coefficients[i] * data[:, i]
    for i in range(len(coefficients)):
        if i == smallestMiIndex:
            mpgEst2 += coefficients[i] * data[:, i].mean()
            continue
        mpgEst2 += coefficients[i] * data[:, i]
    for i in range(len(coefficients)):
        if i == biggestMiIndex:
            mpgEst3 += coefficients[i] * data[:, i].mean()
            continue
        mpgEst3 += coefficients[i] * data[:, i]

    mpgRMSEList = [0, 0, 0]
    for i in range(len(mpgEstList)):
        mpgRMSEList[i] = np.sqrt(
            np.sum(((mpgEstList[i] - mpgOriginal) ** 2) / len(mpgOriginal))
        )

    return mpgEstList, mpgRMSEList


def compareMPGEst(mpgOriginal, mpgEst, i):
    alphabet = np.arange(max(len(mpgOriginal), len(mpgEst)))
    plt.subplot(3, 1, i + 1)
    plt.plot(alphabet, mpgOriginal, "*")
    plt.plot(alphabet, mpgEst, "*")
    plt.ylabel("MPG")
    plt.title(f"Original (azul) vs Estimado {i + 1} (laranja)")


def main():
    plt.close("all")

    # 1
    data = pd.read_excel(
        "/home/tomm/Documents/University/TI2/src/TP1/CarDataset.xlsx"
    )
    dataNp = data.values
    varnames = data.columns.values.tolist()
    _, columns = dataNp.shape  # A folha não pede, mas é útil

    # 2
    plt.figure()
    plt.subplots_adjust(wspace=0.5, hspace=1)
    for i in range(columns - 1):
        compareMPG(dataNp, varnames, i)
    plt.show()

    # 3
    dataNp = dataNp.astype(np.uint16)
    alphabet = np.unique(dataNp)
    print("Alfabeto:\n", alphabet, "\n")

    # 4
    # Nesta lista estarao os arrays com as ocorrencias de cada numero de cada
    # variavel. O array da posicao 0 nesta lista corresponde as ocorrencias da
    # variavel na posicao 0 na varnames ("Acceleration")
    occurrencesList = list()
    for i in range(columns):
        valueList = dataNp[:, i]
        occurrencesList.append(getOccurrences(valueList))

    # 5
    for i in range(len(occurrencesList)):
        showOccurrences(occurrencesList, i, varnames)

    # 6
    # varname: binningSize
    binningDict = {
        "Weight": 40,
        "Displacement": 5,
        "Horsepower": 5,
    }
    for varName, binningSize in binningDict.items():
        varIndex = varnames.index(varName)
        dataNp[:, varIndex] = binning(dataNp[:, varIndex], binningSize)
        occurrencesList[varIndex] = getOccurrences(dataNp[:, varIndex])
        showOccurrences(occurrencesList, varIndex, varnames)

    # 7
    entropyList = list()
    probabilityList = list()
    print("Valores de entropia: ")
    for i in range(len(occurrencesList)):
        entropy, probability = entropyAndProbability(occurrencesList[i])
        entropyList.append(entropy)
        probabilityList.append(probability)
        print(f"{varnames[i]}: {entropy}")
    print()

    # 8
    for i in range(len(occurrencesList)):
        mean, variance = huffmanEnconde(dataNp[:, i], probabilityList[i])
        print(f"{varnames[i]}:")
        print(f"- Valor médio de bits por símbolo: {mean}")
        print(f"- Variância ponderada dos comprimentos: {variance}")
    print()

    # 9
    print("Valores de coeficiente de correlação:")
    for i in range(columns - 1):
        print(f"{varnames[i]} com {varnames[-1]}: {corrcoef(dataNp, i)}")
    print()

    # 10
    print("Informação mútua entre MPG e as restantes variáveis:")
    biggestMiIndex = 0
    biggestMi = 0
    smallestMiIndex = 0
    smallestMi = float("inf")
    for i in range(columns - 1):
        mutualInfoNum = mutualInfo(dataNp, entropyList, i)
        if smallestMi > mutualInfoNum:
            smallestMiIndex = i
            smallestMi = mutualInfoNum
        if biggestMi < mutualInfoNum:
            biggestMiIndex = i
            biggestMi = mutualInfoNum
        print(f"{varnames[i]} com {varnames[-1]}: {mutualInfoNum}")
    print()

    # 11
    print("MPG original VS MPG estimado")
    mpgEstList, mpgRMSEList = estimateMPG(
        dataNp, biggestMiIndex, smallestMiIndex
    )
    # Porque vao haver numeros negativos
    dataNp = dataNp.astype(np.int32)
    plt.figure()
    plt.subplots_adjust(hspace=0.5)
    for i, mpgEst in enumerate(mpgEstList):
        mpgEst = mpgEst.astype(np.int32)
        mpgRMSE = mpgRMSEList[i]
        print(f"MPG estimado {i}")
        # Apenas para comparar o original com os estimados lado a lado
        print("  O", "  E")
        print(
            np.stack(
                (dataNp[:20, -1].transpose(), mpgEst[:20].transpose()), axis=1
            )
        )
        print("...")
        print(
            f"Média da diferença de valores: {(dataNp[:, -1] - mpgEst).mean()}"
        )
        print(f"Root-mean-square error: {mpgRMSE}")
        print()
        compareMPGEst(dataNp[:, -1], mpgEst, i)
    plt.show()


if __name__ == "__main__":
    main()
