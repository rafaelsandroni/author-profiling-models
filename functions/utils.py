import os
import json

def checkFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def listProblems(filter_dataset_name = None, filter_task = None):

    corpus = {
		'b5post':       [['gender','age','religion','it'],                         ['pt']],
		'brmoral':      [['gender','age','religion','education','it','politics'],     ['pt']],
		'esic':		[['gender','age','education','profession','region','city'],  ['pt']],
		'brblogset':	[['gender','age','education'],                          ['pt']],
		'enblog':	[['gender','age'],                                      ['en']],
		'pan13':	[['gender','age'],                                      ['en','es']],
		'smscorpus':	[['gender','age'],                                      ['en']]
    }

    problems = []
    for i in corpus.items():
        dataset_name = i[0]

        if filter_dataset_name != None and filter_dataset_name != dataset_name: continue

        for task in i[1][0]:

            if filter_task != None and filter_task != task: continue

            for lang in i[1][1]:
                problems.append([task, dataset_name, lang])


    return problems

