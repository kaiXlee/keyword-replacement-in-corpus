# Kyle Lee

from pathlib import Path
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np



def loadSlang(fName):
    with open(fName) as file:
        for line in file:
            if(line!="\n"):
                line = line.strip("\n") # strip trailing newline
                temp = line.split("\t")
                slangDict.update({temp[0]: temp[1]})
    return slangDict

def loadCorpusDf(fName):
    df = pd.read_csv(fName)
    texts = df["text"]
    return texts

def processMultipleTexts(texts, slangDict, wordCount=float('inf'), wordsProcessedCount = 0, slangCount = 0):
    for i, text in enumerate(texts):
        if(wordsProcessedCount >= wordCount):
            break
        words = None
        (wordsProcessedCount, slangCount, words) = processStr(text, slangDict, wordCount, wordsProcessedCount, slangCount)
    return wordsProcessedCount, slangCount


def processStr(string, slangDict, wordCount=float('inf'), wordsProcessedCount=0, slangCount=0, verbose=False):
    words = string.split(" ")
    for i, word in enumerate(words):
        if(wordsProcessedCount >= wordCount):
            break
        # key in dictionary has a time complexity of O(1) according to https://wiki.python.org/moin/TimeComplexity
        if word in slangDict:
            slangCount += 1
            words[i] = slangDict[word]
            if verbose:
                print(f"\"{word}\" is replaced with: \"{slangDict[word]}\"")
        wordsProcessedCount += 1
    return wordsProcessedCount, slangCount, words


def tryItABunch(myFunc, texts , slangDict, startN=10, endN=100, stepSize=10, numTrials=20, listMax = 10):
    nValues = [n for n in range(startN, endN, stepSize)]
    timeValues = []
    for n in range(startN, endN, stepSize):
        runtime = 0
        for x in range(0, numTrials):
            startTime = time.time()
            myFunc(texts, slangDict, n)
            endTime = time.time()
            runtime += (endTime - startTime) * 1000 # measured in milliseconds
        timeValues.append(runtime/numTrials)
    return nValues, timeValues


if __name__ == "__main__":
    slangDict = {}
    dirname = os.path.dirname(__file__)
    slangDict = loadSlang(os.path.join(dirname, '../dataset/slangs.txt'))
    texts = loadCorpusDf(os.path.join(dirname, "./../dataset/Android_Q/android-10_bd_2019-01-01-00-00-00_rd_2019-04-22-14-07-31.csv")) # size = 48136
    
    # Sanity Check: we expect text[2] has two matches. "a3" is replaced twice
    (x, y, proccessedStr) = processStr( string = texts[2], slangDict = slangDict, wordCount = 100, verbose = True)
    print("\nORIGINAL: "+texts[2])
    print("\nPROCESSED: "+' '.join(proccessedStr))

    # Plotting asymtotic time complexity
    (nValues, timeValues) = tryItABunch(myFunc = processMultipleTexts, texts = texts, slangDict = slangDict, startN = 10000, endN = 48000, stepSize=1000, numTrials=50)
    plt.plot(nValues, timeValues, color="blue", label="test1")
    z = np.polyfit(nValues, timeValues, 1)
    p = np.poly1d(z)
    plt.plot(nValues,p(nValues),"r--")
    plt.xlabel("n")
    plt.ylabel("Time(ms)")
    plt.legend()
    plt.title("Corpus Transform")
    plt.show()
