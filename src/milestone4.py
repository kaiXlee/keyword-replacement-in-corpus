# Alex Rogov

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
                # strip trailing newline
                line = line.strip("\n")
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
    hints = ["'"]   # " . " can be added but how to differenciate from a word ending in a sentence (eg. someWord.)
    for i, word in enumerate(words):
        replaceEntireWord = True
        if(wordsProcessedCount >= wordCount):
            break
        # key in dictionary has a time complexity of O(1) according to https://wiki.python.org/moin/TimeComplexity

        # Check if word has apostrophy type abbreviation:
        for j in range(len(hints)):
            if hints[j] in word:
                replaceEntireWord = False
                index = word.index(hints[j])
                substringShort = word[index:] # " 'll, 've, 're "
                substringLong = word[index-1:]  # "n't"
                #HashMap of abbreviations to full word. (eg. key="'t" and value="not")
                if substringShort in slangDict:
                    slangCount += 1
                    words[i] = word.replace(substringShort, " " + slangDict[substringShort])
                    if verbose:
                        print(f"\"{word}\" is replaced with: \"{words[i]}\"")
                elif substringLong in slangDict:
                    slangCount += 1
                    words[i] = word.replace(substringLong, " " + slangDict[substringLong])
                    if verbose:
                        print(f"\"{word}\" is replaced with: \"{words[i]}\"")
                wordsProcessedCount += 1

        # Check if entire word shoud be replaced
        if replaceEntireWord and word in slangDict:
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
            runtime += (endTime - startTime) * 1000 # scale to milliseconds
            # print(runtime)
        timeValues.append(runtime/numTrials)
    return nValues, timeValues

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    texts = loadCorpusDf(os.path.join(dirname, "./../dataset/Android_Q/android-10_bd_2019-01-01-00-00-00_rd_2019-04-22-14-07-31.csv")) # size = 48136
    
    slangDict = {}
    # Choose one of each files to use as a dictionary
    slangsContractionsFile = '../dataset/slangs_and_contractions.txt'
    slangsFile = '../dataset/slangs.txt'
    chosenFile = os.path.join(dirname, slangsFile)
    slangDict = loadSlang(chosenFile)

    # Check for contractions
    (x, y, proccessedStr) = processStr( string = texts[10], slangDict = slangDict, wordCount = 100, verbose=True)
    print("\nORIGINAL: "+texts[10])
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
    plt.title("Corpus Transform\n" + os.path.basename(chosenFile))
    plt.show()