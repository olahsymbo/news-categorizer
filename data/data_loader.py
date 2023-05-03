import os
import sys
import inspect

app_path = inspect.getfile(inspect.currentframe())
cate_dir = os.path.realpath(os.path.dirname(app_path))

import numpy as np
import pandas as pd
import string


class NewsDataLoader:
  
  def __init__(self, source_dir):
    self.source_dir = source_dir
    
  def load_all_news(self):
    datan = []
    entertain = os.listdir(os.chdir(os.path.join(cate_dir, self.source_dir, 'entertainment')))

    files_entertain = []
    target_entertain = np.array([1] * len(entertain))
    for sample in entertain:
        sl = open(sample, 'r').read()
        slnn = sl.split()
        table = str.maketrans('', '', string.punctuation)
        stripped = sl.translate(table)
        files_entertain.append(stripped)

    sport = os.listdir(os.chdir(os.path.join(cate_dir, self.source_dir, 'sport')))

    files_sport = []
    target_sport = np.array([2] * len(sport))
    for sample in sport:
        spr = open(sample, 'r').read()
        sprn = spr.split()
        table = str.maketrans('', '', string.punctuation)
        sport_stripped = spr.translate(table)
        files_sport.append(sport_stripped)

    politics = os.listdir(os.chdir(os.path.join(cate_dir, self.source_dir, 'politics')))

    files_politics = []
    target_politics = np.array([3] * len(politics))
    for sample in politics:
        pol = open(sample, 'r').read()
        poln = pol.split()
        table = str.maketrans('', '', string.punctuation)
        pol_stripped = pol.translate(table)
        files_politics.append(pol_stripped)

    business = os.listdir(os.chdir(os.path.join(cate_dir, self.source_dir, 'business')))

    files_business = []
    target_business = np.array([4] * len(business))
    for sample in business:
        buss = open(sample, 'r').read()
        bussn = buss.split()
        table = str.maketrans('', '', string.punctuation)
        buss_stripped = buss.translate(table)
        files_business.append(buss_stripped)

    tech = os.listdir(os.chdir(os.path.join(cate_dir, self.source_dir, 'tech')))

    files_tech = []
    target_tech = np.array([5] * len(tech))
    for sample in tech:
        tec = open(sample, 'r').read()
        tecn = tec.split()
        table = str.maketrans('', '', string.punctuation)
        tec_stripped = tec.translate(table)
        files_tech.append(tec_stripped)

    
    datan.extend(files_entertain)
    datan.extend(files_sport)
    datan.extend(files_politics)
    datan.extend(files_business)
    datan.extend(files_tech)
    
    return datan
