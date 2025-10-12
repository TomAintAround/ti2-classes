from scipy.io import wavfile
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np


def apresentarInfo(nomeFicheiro: str, fs: int, nrBitsQuant: int) -> None:
    print("Informções do ficheiro")
    print(f"Nome: {nomeFicheiro}")
    print(f"Taxa de amostragem: {fs / 1000:.1f} kHz")
    print(f"Quantização: {nrBitsQuant} bits")


def play(data: np.ndarray, fs: int) -> None:
    sd.play(data, fs)
    sd.wait()


def visualizacaoGrafica(*args) -> None:
    data: np.ndarray = args[0]
    fs: int = args[1]
    ts: float = 1 / fs
    tini: float = 0.0
    tfim: float = (len(data) - 1) * ts
    if len(args) >= 3:
        tini = args[2]
    if len(args) >= 4:
        tfim = args[3]

    normalData: np.ndarray = data[:, :].astype("float64")
    normalData /= 2 ** (data.itemsize * 8 - 1)
    tempo: np.ndarray = np.arange(0, len(normalData)) * ts
    canalEsq: np.ndarray = normalData[:, 0]
    canalDir: np.ndarray = normalData[:, 1]
    window: tuple[float, float, float, float] = (tini, tfim, -1.0, 1.0)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(tempo, canalEsq)
    plt.axis(window)
    plt.xlabel("Tempo [s]")
    plt.ylabel("Amplitude [-1:1]")
    plt.title("Canal Esquerdo")
    plt.subplot(2, 1, 2)
    plt.plot(tempo, canalDir)
    plt.axis(window)
    plt.xlabel("Tempo [s]")
    plt.ylabel("Amplitude [-1:1]")
    plt.title("Canal Direito")
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def adicionarRuido(data: np.ndarray, amp: int) -> np.ndarray:
    dataHeight, dataWidth = data.shape
    dataRuido: np.ndarray = (
        np.random.rand(dataHeight, dataWidth) * amp * 2 - amp
    )
    dataRuido = dataRuido.astype("int16")
    dataRuido += data
    valorMax: int = 2 ** (data.itemsize * 8 - 1)
    dataRuido[dataRuido < -valorMax] = -valorMax
    dataRuido[dataRuido >= valorMax] = valorMax - 1

    return dataRuido


def guitarraDireito(
    data: np.ndarray, fichGuitarra: str, tini: int
) -> np.ndarray:
    _, dataGuitarra = wavfile.read(f"src/TP0/Sound/{fichGuitarra}")
    dataGuitarra: np.ndarray = dataGuitarra[tini:]
    guitHeight, width = dataGuitarra.shape
    dataCopy: np.ndarray = data[tini:]
    newData: np.ndarray = np.zeros_like(dataCopy[tini:])
    newHeight, _ = newData.shape
    if guitHeight > newHeight:
        zerosConcat: np.ndarray = np.zeros([guitHeight - newHeight, width])
        dataCopy = np.concatenate((dataCopy, zerosConcat), axis=0)
        newData = np.concatenate((newData, zerosConcat), axis=0)
    elif guitHeight < newHeight:
        zerosConcat: np.ndarray = np.zeros([newHeight - guitHeight, width])
        dataGuitarra = np.concatenate((dataGuitarra, zerosConcat), axis=0)

    newData[:, 0] = dataCopy[:, 0]
    newData[:, 1] = dataGuitarra[:, 0]

    return newData


def contornoAmplitude(data: np.ndarray, W: int) -> np.ndarray:
    dataCopy: np.ndarray = data[:]
    dataCopy[dataCopy < 0] = 0
    dataContorno: np.ndarray = np.zeros_like(dataCopy)
    amp: int = np.floor(W / 2)
    for i in range(len(dataCopy)):
        indexMin = int(max(i - amp, 0))
        indexMax = int(min(i + amp, len(dataCopy)))
        dataContorno[i] = np.mean(dataCopy[indexMin:indexMax])

    return dataContorno


def main() -> None:
    filename: str = "drumloop.wav"
    fs, data = wavfile.read(f"src/TP0/Sound/{filename}")
    apresentarInfo(filename, fs, data.itemsize * 8)

    visualizacaoGrafica(data, fs)
    play(data, fs)
    play(data, fs * 2)
    play(data, fs / 2)

    ruidoData: np.ndarray = adicionarRuido(data, 15000)
    visualizacaoGrafica(ruidoData, fs)
    play(ruidoData, fs)

    comGuitarra: np.ndarray = guitarraDireito(data, "guitar.wav", 0)
    visualizacaoGrafica(comGuitarra, fs)
    play(comGuitarra, fs)

    dataContorno: np.ndarray = contornoAmplitude(data, 10)
    visualizacaoGrafica(dataContorno, fs)
    play(dataContorno, fs)


if __name__ == "__main__":
    main()
