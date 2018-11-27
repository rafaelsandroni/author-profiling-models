import os
import json

def checkFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def listProblems():

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
        for task in i[1][0]:
            for lang in i[1][1]:
                problems.append([task, dataset_name, lang])


    return problems

