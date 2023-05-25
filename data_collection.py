# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:02:11 2023

@author: frank
"""

import glassdoor_scraper as gs
import pandas as pd

path="C:\Program Files (x86)\Google\chrome-driver\chromedriver.exe"

df = gs.get_jobs('data scientist', 15, True, path, 6)
df

df = pd.read_csv('glassdoor_jobs.csv')
df