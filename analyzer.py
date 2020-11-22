import json
import operator
import math
import requests
import numpy as np
from pymongo.mongo_client import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure
from matplotlib import pyplot as plt
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sentiment_intensity_analyzer = SentimentIntensityAnalyzer()
import pandas as pd
import os


'''

def download_file(filename):
    file = filename
    url = 'https://wikitalkpages.s3.ap-south-1.amazonaws.com/' + file + '.json'
    r = requests.get(url, allow_redirects=True)
    open(file + '.json', 'wb').write(r.content)


def putInDatabase(collection_name, file_name, myclient, mongoClientDB):
    collection = mongoClientDB[collection_name]
    json_file = open(file_name)
    data = json_file.read().strip("[]").split("\"},")
    for i in range(len(data)):
        if i != len(data)-1:
            data[i] += "\"}"
        ins = json.loads(data[i])
        collection.insert(ins)

'''


class Analyzer:

    def __init__(self, myclient: MongoClient, mongoClientDB: Database):
        self.myclient = myclient
        self.mongoClientDB = mongoClientDB
        self.dataCollectionName = None

    def download_file(self, filename):
        '''   
        The function downloads the full article dataset from the server

        Parameters
        ----------
        filename : TYPE
            DESCRIPTION.
        Returns
        -------
        None.
        '''

        fn = filename
        url = 'https://wikitalkpages.s3.ap-south-1.amazonaws.com/' + fn + '.json'
        r = requests.get(url, allow_redirects=True)
        json_file = fn + '.json'
        open(filename, 'wb').write(r.content)

    def putInDatabase(self, folder_path, file_name, collection_name,encoding="utf-8"):
        if collection_name in (self.mongoClientDB).list_collection_names():
            collection = self.mongoClientDB[collection_name]
            collection.drop()
        collection = self.mongoClientDB[collection_name]
        file_path = folder_path + file_name
        with open(file_path, encoding=encoding) as fh:
            file_data = json.load(fh)
        if isinstance(file_data, list):
            collection.insert_many(file_data)
        else:
            collection.insert_one(file_data)

    def listOfEditors(self, collection_name):
        collection = self.mongoClientDB[collection_name]
        editors = (collection.distinct('user'))
        return editors

    def topNEditors(self, collection_name, n):
        top_editors = {}
        collection = self.mongoClientDB[collection_name]
        result = list(collection.aggregate(
            [{"$group": {"_id": "$user", "num_comments": {"$sum": 1}}}]))
        for each in result:
            editor = each.get('_id')
            num_comments = each.get('num_comments')
            top_editors[editor] = num_comments
        if n > len(top_editors):
            top_editors = list(top_editors.items())
        else:
            top = Counter(top_editors)
            top_editors = top.most_common(n)
        return top_editors

    def getCommentSentiments(self, collection_name, user=None):
        '''
        Find sentiments of comments in the specified collection.
        If `user` parameter is unspecified or None, predicts the sentiments of all the comments
        in the specified collection.
        Positive sentiment : (compound score >= 0.05)
        Neutral sentiment : (compound score > -0.05) and (compound score < 0.05)
        Negative sentiment : (compound score <= -0.05)
        '''
        collection = Collection(self.mongoClientDB, collection_name)
        comments = list()

        if user == None:
            for comment in collection.find():
                comment["polarity_scores"] = sentiment_intensity_analyzer.polarity_scores(comment["text"])
                comments.append(comment)
        else:
            for comment in collection.find({"user":user}):
                comment["polarity_scores"] = sentiment_intensity_analyzer.polarity_scores(comment["text"])
                comments.append(comment)

        return comments

    def getCommentsByDate(self, collection_name, day=None, month=None, year=None):
        collection = self.mongoClientDB[collection_name]
        comments = []
        for each in collection.find():
            date = each["date"]
            df = pd.DataFrame({'date': [date]})
            df["date"] = pd.to_datetime(df["date"])
            df["month"] = df["date"].dt.month
            df["year"] = df["date"].dt.year
            df["day"] = df["date"].dt.day
            day_match = True
            month_match = True
            year_match = True
            if not day == None:
                if not (df["day"] == day).all():
                    day_match = False
            if not month == None:
                if not (df["month"] == month).all():
                    month_match = False
            if not year == None:
                if not (df["year"] == year).all():
                    year_match = False
            if day_match and month_match and year_match:
                comments.append(each)
        return comments

    def getCommentsForGivenDuration(self, collection_name, start_date, end_date):
        collection = self.mongoClientDB[collection_name]
        comments = []
        for each in collection.find():
            date = each["date"]
            date = date.split('T')[0]
            date = pd.to_datetime(date)
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            if date >= start_date and date <= end_date:
                comments.append(each)
        return comments


    def commonEditors(self, collection_name1, collection_name2):
        collection1 = self.mongoClientDB[collection_name1]
        editors1 = (collection1.distinct('user'))
        collection2 = self.mongoClientDB[collection_name2]
        editors2 = (collection2.distinct('user'))
        common_editors = list(set(editors1) & set(editors2))
        return common_editors

    """def deleteCollection(self, collection_name):
        collection = self.mongoClientDB[collection_name]
        collection.drop()

    def downloadAndLoad(self, collection_name, filename):
        self.download_file(filename)
        self.putInDatabase(collection_name, filename)

    def setCollectionName(self, dataCollectionName):
        self.dataCollectionName = dataCollectionName

    def getAlldata(self):
        mycol = self.mongoClientDB[self.dataCollectionName]
        arr = []
        for x in mycol.find():
            arr.append(x)
        return arr

    def totalNumberOfComments(self):
        mycol = self.mongoClientDB[self.dataCollectionName]
        return mycol.count()

    def getAllAuthors(self):
        mycol = self.mongoClientDB[self.dataCollectionName]
        return mycol.distinct('user')

    def allAuthorsContribution(self):
        mycol = self.mongoClientDB[self.dataCollectionName]
        pipeline = {"$group": {"_id": "$user", "count": {"$sum": 1}}}
        dictionary = mycol.aggregate([pipeline])
        answer = {}
        for item in dictionary:
            answer[item["_id"]] = item["count"]
        return answer

    def getTopNContributors(self, n):
        authors = self.allAuthorsContribution()
        sorted_d = dict(
            sorted(authors.items(), key=operator.itemgetter(1), reverse=True))
        new_dict = {}
        num = 0
        for key, val in sorted_d.items():
            if num == n:
                break
            num += 1
            new_dict[key] = val
        return new_dict

    def getLeastNContributors(self, n):
        authors = self.allAuthorsContribution()
        sorted_d = dict(
            sorted(authors.items(), key=operator.itemgetter(1), reverse=False))
        new_dict = {}
        num = 0
        for key, val in sorted_d.items():
            if num == n:
                break
            num += 1
            new_dict[key] = val
        return new_dict

    def allCommentStatistics(self):
        dictionary = {}
        min_len = 999999999999999  # Done
        max_len = 0  # Done
        avg = 0  # Done
        totalLen = 0  # Done
        standardDev = 0  # Done
        count = 0  # Done
        variance = 0  # Done

        all_data = self.getAlldata()

        for item in all_data:
            count += 1
            x = len(item['text'])
            if x < min_len:
                min_len = x
            if x > max_len:
                max_len = x
            totalLen += x

        avg = float(totalLen) / float(count)
        value = 0
        for item in all_data:
            x = len(item['text'])
            value += (x - avg) * (x - avg)
        variance = float(value) / float(count)
        standardDev = math.sqrt(variance)

        dictionary['min_len'] = min_len
        dictionary['max_len'] = max_len
        dictionary['avg'] = avg
        dictionary['totalLen'] = totalLen
        dictionary['standardDev'] = standardDev
        dictionary['count'] = count
        dictionary['variance'] = variance
        return dictionary

    def showCommentsStatistics(self, numBucket):
        dictionary = self.allCommentStatistics()
        numberOfBuckets = float(numBucket)
        bucketLength = float(
            dictionary['max_len'] - dictionary['min_len']) / numBucket
        blocks = [0 for k in range(numBucket+1)]
        all_data = self.getAlldata()
        min_len = dictionary['min_len']

        for item in all_data:
            x = len(item['text'])
            blocks[int((x - min_len)/bucketLength)] += 1

        plt.xlabel('Length of comment')
        plt.ylabel('Comment Length Frequency')

        plt.bar([i+1 for i in range(numBucket+1)], blocks,
                width=0.8, bottom=None, align='center')
        plt.show()

    def getAllRevisionIds(self):
        mycol = self.mongoClientDB[self.dataCollectionName]
        return mycol.distinct('revision_id')

    def commentDictionaryRevisionId(self):
        rev_id = self.getAllRevisionIds()
        dictionary = {}

        for item in rev_id:
            dictionary[item] = []
        all_data = self.getAlldata()

        for item in all_data:
            dictionary[item['revision_id']].append(item)

        return dictionary

    def commentCountByRevisionId(self):
        dictionary = self.commentDictionaryRevisionId()
        for key, val in dictionary.items():
            dictionary[key] = len(val)
        return dictionary

    def getTopNRevisions(self, n):
        authors = self.commentCountByRevisionId()
        sorted_d = dict(
            sorted(authors.items(), key=operator.itemgetter(1), reverse=True))
        new_dict = {}
        num = 0
        for key, val in sorted_d.items():
            if num == n:
                break
            num += 1
            new_dict[key] = val
        return new_dict

    def getLeastNRevisions(self, n):
        authors = self.commentCountByRevisionId()
        sorted_d = dict(
            sorted(authors.items(), key=operator.itemgetter(1), reverse=False))
        new_dict = {}
        num = 0
        for key, val in sorted_d.items():
            if num == n:
                break
            num += 1
            new_dict[key] = val
        return new_dict

    def commentsFilterByRevisionId(self, revisionId):
        mycol = self.mongoClientDB[self.dataCollectionName]
        rev_id_comments = mycol.find({"revision_id": revisionId})
        return list(rev_id_comments)

    def getDepthOfCommentsInRevision(self, revisionId):
        arr = self.commentsFilterByRevisionId(revisionId)
        dictionary = {}

        for item in arr:
            dictionary[item['id']] = item['parent_id']

        keys = list(dictionary.keys())
        depth_dict = {}

        for key, val in dictionary.items():
            depth_dict[key] = 1

        for item in keys:
            it = item
            while dictionary[it] != 0:
                depth_dict[item] += 1
                it = dictionary[it]
        return depth_dict

    def getDepthOfComment(self, id, revisionId):
        arr = self.commentsFilterByRevisionId(revisionId)
        dictionary = {}

        for item in arr:
            dictionary[item['id']] = item['parent_id']

        keys = list(dictionary.keys())
        depth_dict = {}

        for key, val in dictionary.items():
            depth_dict[key] = 1

        for item in keys:
            it = item
            while dictionary[it] != 0:
                depth_dict[item] += 1
                it = dictionary[it]
        print(depth_dict)
        return depth_dict[id]

    def depthStatisticsByRevisionId(self, revisionId):
        dictionary = self.getDepthOfCommentsInRevision(revisionId)
        arr = []

        for key, val in dictionary.items():
            arr.append(val)

        dictionary = {}
        min_len = 999999999999999  # Done
        max_len = 0  # Done
        avg = 0  # Done
        totalLen = 0  # Done
        standardDev = 0  # Done
        count = len(arr)  # Done
        variance = 0  # Done

        for item in arr:
            if item < min_len:
                min_len = item
            if item > max_len:
                max_len = item
            totalLen += item

        avg = float(totalLen) / float(count)

        for item in arr:
            variance += (item - avg) * (item - avg)

        variance = float(variance) / float(count)
        standardDev = math.sqrt(variance)

        dictionary['min_len'] = min_len
        dictionary['max_len'] = max_len
        dictionary['avg'] = avg
        dictionary['standardDev'] = standardDev
        dictionary['count'] = count
        dictionary['variance'] = variance
        return dictionary

    def getAllSections(self):
        mycol = self.mongoClientDB[self.dataCollectionName]
        return mycol.distinct('section')

    def commentsFilterBySection(self, sectionName):
        mycol = self.mongoClientDB[self.dataCollectionName]
        return list(mycol.find({"section": sectionName}))

    def getSectionwiseCommentCount(self):
        mycol = self.mongoClientDB[self.dataCollectionName]
        pipeline = {"$group": {"_id": "$section", "count": {"$sum": 1}}}
        dictionary = mycol.aggregate([pipeline])
        answer = {}
        for item in dictionary:
            answer[item["_id"]] = item["count"]
        return answer

    def showSectionsStatistics(self):
        dictionary = self.getSectionwiseCommentCount()

        arr = []
        for key, val in dictionary.items():
            arr.append(val)

        dictionary = {}
        min_len = 999999999999999  # Done
        max_len = 0  # Done
        avg = 0  # Done
        totalLen = 0  # Done
        standardDev = 0  # Done
        count = len(arr)  # Done
        variance = 0  # Done

        for item in arr:
            if item < min_len:
                min_len = item
            if item > max_len:
                max_len = item
            totalLen += item

        avg = float(totalLen) / float(count)

        for item in arr:
            variance += (item - avg) * (item - avg)

        variance = float(variance) / float(count)
        standardDev = math.sqrt(variance)

        dictionary['min_len'] = min_len
        dictionary['max_len'] = max_len
        dictionary['avg'] = avg
        dictionary['standardDev'] = standardDev
        dictionary['count'] = count
        dictionary['variance'] = variance
        return dictionary"""


if __name__ == '__main__':

    myclient = MongoClient("mongodb://localhost:27017/")

    try:
        myclient.admin.command('ismaster')
        print("\nServer available\n")
    except ConnectionFailure:
        print("\nServer not available\n")
        exit()

    mongoClientTalkPagesDB = myclient['mywikidumptalkpages']
    analyzer_talk = Analyzer(myclient, mongoClientTalkPagesDB)

    mongoClientRevisionsDB = myclient['mywikidumprevisions']
    analyzer_revision = Analyzer(myclient, mongoClientRevisionsDB)
    # analyzer.putInDatabase('sample.json')
    for filename in os.listdir('Talk Pages'):
        if filename.endswith('.json'):
            collection_name = filename[:-5]
            analyzer_talk.putInDatabase('Talk Pages/', filename, collection_name)

    for filename in os.listdir('Revision'):
        if filename.endswith('.json'):
            collection_name = filename[:-5]
            analyzer_revision.putInDatabase('Revision/', filename, collection_name)

    # List the editors
    collection_name = 'India'
    if collection_name not in (mongoClientTalkPagesDB).list_collection_names():
        print("\n--Collection with specified name does not exist--\n")
    else:
        editors = analyzer_talk.listOfEditors(collection_name)
        print('\n\'', collection_name, '\' talk page has', len(editors), 'editors\n')

    # Find top editors
    collection_name = 'India'
    if collection_name not in (mongoClientTalkPagesDB).list_collection_names():
        print("\n--Collection with specified name does not exist--\n")
    else:
        n = 5
        top_editors = analyzer_talk.topNEditors(collection_name, n)
        print('\nList of top', len(top_editors), 'editors :', top_editors, '\n')

    # Find the sentiments of each comment
    collection_name = 'India'
    if collection_name not in (mongoClientTalkPagesDB).list_collection_names():
        print("\n--Collection with specified name does not exist--\n")
    else:
        comments = analyzer_talk.getCommentSentiments(collection_name, "RegentsPark")
        for comment in comments:
            print('\nPolarity scores of the comment with id', comment["id"], ':', comment["polarity_scores"], '\n')

    # Find comments by day/month/year
    collection_name = 'India'
    if collection_name not in (mongoClientTalkPagesDB).list_collection_names():
        print("\n--Collection with specified name does not exist--\n")
    else:
        comments = analyzer_talk.getCommentsByDate(collection_name, day=5, month=11, year=2020)
        print('\nComments for specified date:\n', comments, '\n')
    
    # Find comments for given duration
    collection_name = 'India'
    if collection_name not in (mongoClientTalkPagesDB).list_collection_names():
        print("\n--Collection with specified name does not exist--\n")
    else:
        date1 = '2020-11-04'
        date2 = '2020-11-15'
        comments = analyzer_talk.getCommentsForGivenDuration(collection_name, date1, date2)
        print('\nComments of given duration:\n', comments, '\n')

    # Common editors
    collection_name1 = 'India'
    collection_name2 = 'Narendra Modi'
    if collection_name1 not in (mongoClientTalkPagesDB).list_collection_names() or collection_name2 not in (mongoClientTalkPagesDB).list_collection_names():
        print("\n--Collection with specified name does not exist--\n")
    else:
        common_editors = analyzer_talk.commonEditors(
            collection_name1, collection_name2)
        print('\nCommon editors are :', common_editors, '\n')

    # analyzer.setCollectionName('sample')
    # print(analyzer.getAllAuthors())
    # analyzer.downloadAndLoad(
        # 'Indian_Institute_of_Technology_Ropar', 'Indian_Institute_of_Technology_Ropar')

    """
	analyzer.deleteCollection('Indian_Institute_of_Technology_Ropar')
	analyzer.deleteCollection('Animal')
	analyzer.deleteCollection('Taj_Mahal')
	analyzer.deleteCollection('United_States')
	analyzer.deleteCollection('World_Wide_Web')
	analyzer.deleteCollection('mywikicollection')
	"""
