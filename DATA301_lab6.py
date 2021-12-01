import glob
import nltk
import os
import math
import pandas as pd
import numpy as np
from nltk.corpus import stopwords

def getTFIDFFrame(filepath, useQueryFormula):
    documentFilePaths = glob.glob(filepath + '/*.txt')
    # documentFilePaths = glob.glob("labs/lab6/documents/"+'*.txt')

    # import all documents in document folder
    docs = []
    for file in documentFilePaths:
        with open(file,'r') as f:
            docID = os.path.basename(file.split('.')[0])
            docs.append({'document':docID,'text':f.read()})    

    # mapping documentID to the raw contents of said document
    rawDocumentContents = pd.DataFrame(docs)
    rawDocumentContents.set_index('document')


    # "massage" the text to be without '.,!-= etc'
    rawDocumentContents['words'] = rawDocumentContents['text'].str.strip().str.split('[\W]+')


    # create a mapping of documents -> word in document to perform groupby operations on
    stop_words = list(stopwords.words('english'))
    wordToDoc = []

    for i in range(0,len(rawDocumentContents)):
        for word in rawDocumentContents.iloc[i]['words']:
            if(word.lower() not in stop_words and word != ''):
                wordToDoc.append((rawDocumentContents.iloc[i]['document'], word.lower()))
    wordsInDocs = pd.DataFrame(wordToDoc, columns=['document', 'word'])

    # count the number of each word per document
    counts = wordsInDocs.groupby('document')['word'].value_counts().to_frame().rename(columns={'word':'frequency'})

    # count the max frequency of any word per document
    maxFrequency = counts.groupby('document').max().rename(columns={'frequency':'maxFr'})

    # extrapolate maxFrequency over each wordCount to calculate tf
    outputFrame = counts.join(maxFrequency)
    if(useQueryFormula):
        outputFrame['tf']=(0.5+0.5*outputFrame['frequency']/outputFrame['maxFr'])
    else:
        outputFrame['tf']=outputFrame['frequency']/outputFrame['maxFr']

    # print(wordsInDocs)
    numDocuments = len(rawDocumentContents)

    # calculate the number of documents each word appears in
    docFrequency = wordsInDocs.groupby('word')['document'].nunique().to_frame().rename(columns={'document':'df'})

    # calculate the inverse document frequency
    docFrequency['idf'] = np.log2(numDocuments/docFrequency['df'].values)

    # join two frames on word index; for each document, add df and idf for each word
    # only keeps those with significant tfidf (> 0?)
    outputFrame = outputFrame.join(docFrequency)

    # calculate tfidf
    outputFrame['tfidf']=outputFrame['tf']*outputFrame['idf']

    return outputFrame

def getSimilaritiesFrame(docFrame, queryFrame):
    # merge to only keep words that are in both documents and queries
    mergedFrame = pd.merge(docFrame, queryFrame, left_index=True, right_index=True)

    # for each query, calculate cos similarity between tfidf_x and tfidf_y
    # return dataframe with index = 'query' 'doc': cosSimilarity
    groups = mergedFrame.groupby(['queryID', 'docID'])

    queryMagnitudes = {}
    for grp in queryFrame.groupby('queryID'):
        # print(grp)
        queryMagnitudes[grp[0]] = magnitude(grp[1]['tfidf'])

    queryMagnitudes = pd.Series(queryMagnitudes)
    # print(queryMagnitudes)

    docMagnitudes = {}
    for grp in docFrame.groupby('docID'):
        # docMagnitudes += magnitude(grp['tfidf'])
        # print(grp[1])
        docMagnitudes[grp[0]] = magnitude(grp[1]['tfidf'])
    docMagnitudes = pd.Series(docMagnitudes)
    # print(docMagnitudes)

    similarityFrameData = []

    for groupContainer in groups:
        docID = groupContainer[0][1]
        queryID = groupContainer[0][0]
        groupFrame = groupContainer[1]

        # when there is only one word shared between the two documents, cosSimilarity always equals 1, which throws it off
        # is this the right formula? am I joining the data correctly?
        cosSimilarity = groupFrame['tfidf_x'].dot(groupFrame['tfidf_y'])/(queryMagnitudes[queryID] * docMagnitudes[docID])
        similarityFrameData.append({'queryID':queryID,'docID':docID,'cosSimilarity':cosSimilarity})

    sframe = pd.DataFrame(similarityFrameData, columns=['docID', 'queryID', 'cosSimilarity'])
    sframe['docID'] = sframe['docID'].apply(int)
    sframe['queryID'] = sframe['queryID'].apply(int)
    return sframe

def get20MostSimilarPerQuery(similarityFrame):

    queryGroups = similarityFrame.groupby('queryID')
    mostSimilarDocsPerQuery = pd.DataFrame(columns=['queryID', 'docID'])
    for queryGroup in queryGroups:
        queryID = queryGroup[0]
        sorted = queryGroup[1].sort_values(by='cosSimilarity', ascending=False)
        mostSimilarDocs = sorted.set_index('queryID')[['docID', 'cosSimilarity']].iloc[0:20].reset_index()
        # print(mostSimilarDocs)
        mostSimilarDocsPerQuery = mostSimilarDocsPerQuery.append(mostSimilarDocs)

    return mostSimilarDocsPerQuery.set_index(['queryID'])

def magnitude(vector):
    return ((vector**2).sum())**0.5 

def getHumanRelevance(judgementFile):
    data = pd.read_csv(judgementFile,sep=' ')
    data.drop(columns='empty', inplace=True)
    data = data[data['relevance'] < 4]
    data = data[data['relevance'] > 0]

    return data[['queryNum', 'docNum']].rename(columns={'queryNum':'queryID', 'docNum':'docID'}).set_index('queryID')

def calcMAPScore(humanList, computerList):
    if(len(humanList) == 0):
        print('no relevant docs')
        return 0

    numShared = 0
    mapScore = 0
    for i, docNum in enumerate(computerList):
        if docNum in humanList:
            print(f'sharing {docNum}')
            numShared += 1
            mapScore += numShared/(i+1)

    return mapScore/len(humanList)

def calcAllMAPScores(humanJudgement, compJudgement):
    # print(humanJudgement)
    mapScores = []
    humanJudgement = humanJudgement.reset_index()
    compJudgement = compJudgement.reset_index()

    # for query in set(compJudgement['queryID']):
    for query in range(1, 21):
        humanList = humanJudgement[humanJudgement['queryID'] == int(query)]['docID']    # index of queries in humanJudgement are strings instead of ints?
        compList = compJudgement[compJudgement['queryID'] == str(query)]['docID']
        mapScores.append({'queryID':query,'mapScore':(calcMAPScore(humanList,compList))})

    return pd.DataFrame(mapScores)


# nltk.download('stopwords')
# docFrame = getTFIDFFrame('labs/lab6/testFolder', False)
# queryFrame = getTFIDFFrame('labs/lab6/testQueries', True)

# docFrame = getTFIDFFrame('labs/lab6/documents', False)
# queryFrame = getTFIDFFrame('labs/lab6/queries', True)

# queryFrame.index.names = ['queryID', 'word']
# docFrame.index.names = ['docID', 'word']

# similarityFrame = getSimilaritiesFrame(docFrame, queryFrame)
# # print(similarityFrame.head(20))
# computerRelevance = get20MostSimilarPerQuery(similarityFrame)

# print(computerRelevance.head(20))

# print(similarityFrame)

# humanRelevance = getHumanRelevance('labs/lab6/humanJudgementTest')
humanRelevance = getHumanRelevance('labs/lab6/human_judgement.txt')

print(humanRelevance.head(30))

# mapScores = calcAllMAPScores(humanRelevance, computerRelevance)

# print(mapScores)