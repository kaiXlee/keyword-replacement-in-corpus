from pathlib import Path
import os
import pandas as pd
import time
import matplotlib.pyplot as plt



def loadSlang(fName):
    with open(fName) as file:
        for line in file:
            if(line!="\n"):
                # strip trailing newline
                line = line.strip("\n")
                temp = line.split("\t")
                slangDict.update({temp[0]: temp[1]})
    return slangDict

def loadCorpusDf(fName):
    df = pd.read_csv(fName)
    # print(df.head())
    texts = df["text"]
    # for text in texts:
        # print(text)
    return texts

def processMultipleTexts(texts, slangDict, wordCount=float('inf'), wordsProcessedCount = 0, slangCount = 0):
    for i, text in enumerate(texts):
        if(wordsProcessedCount >= wordCount):
            break
        (wordsProcessedCount, slangCount) = processStr(text, slangDict, wordCount, wordsProcessedCount, slangCount)
    # print("wordsProcessedCount: "+str(wordsProcessedCount))
    return wordsProcessedCount, slangCount


def processStr(string, slangDict, wordCount=float('inf'), wordsProcessedCount=0, slangCount=0):
    words = string.split(" ")
    for i, word in enumerate(words):
        if(wordsProcessedCount >= wordCount):
            break
        # key in dictionary has a time complexity of O(1) according to https://wiki.python.org/moin/TimeComplexity
        if word in slangDict:
            slangCount += 1
        wordsProcessedCount += 1
    return wordsProcessedCount, slangCount


def tryItABunch(myFunc, texts , slangDict, startN=10, endN=100, stepSize=10, numTrials=20, listMax = 10):
    nValues = []
    timeValues = []
    for n in range(startN, endN, stepSize):
        startTime = time.time()
        myFunc(texts, slangDict, n)
        endTime = time.time()
        runtime = (endTime - startTime) * 1000 # measured in milliseconds
        nValues.append(n)
        timeValues.append(runtime)
    return nValues, timeValues


if __name__ == "__main__":
    slangDict = {}
    slangDict = loadSlang("../dataset/slangs.txt")
    texts = loadCorpusDf("../dataset/Android_Q/android-10_bd_2019-01-01-00-00-00_rd_2019-04-22-14-07-31.csv") # size = 48136
    # processMultipleTexts(texts, slangDict, 1000)

    # # text[2] has two matches. a3 is matched twice
    # processStr( string = texts[2], slangDict = slangDict, wordCount = 100)
    (nValues, timeValues) = tryItABunch(myFunc = processMultipleTexts, texts = texts, slangDict = slangDict, startN = 10000, endN = 48000, stepSize=1000)

    plt.plot(nValues, timeValues, color="blue", label="test1")
    plt.xlabel("n")
    plt.ylabel("Time(ms)")
    plt.legend()
    plt.title("Corpus Transform")
    plt.show()
