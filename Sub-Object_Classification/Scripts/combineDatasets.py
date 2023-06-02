import pandas as pd
import numpy as np

import argparse


def prepareSubtypes(path):
    df = pd.read_csv(path)
    df.head()
    df.drop([0, 1], axis=0, inplace=True)
    df.drop(df[df['Glon'].isna() | df['Glat'].isna()].index, inplace=True)
    df.sort_values(by=['Spectral Type', 'Sp type', 'Mag type'], inplace=True)
    df.drop_duplicates(subset='LAMOST', inplace=True)

    df.rename(columns={'Class': 'Cl', 'LAMOST': 'Name'}, inplace=True)

    df['root'] = 'subtypes'
    df["plate"] = np.nan
    df["path"] = np.nan
    df["dx"] = np.zeros(df.shape[0])
    df["dy"] = np.zeros(df.shape[0])
    df[['_RAJ2000', '_DEJ2000']] = df[['RAJ2000', 'DEJ2000']].astype(float)
    return df


def prepareData(path):
    data = pd.read_csv(path)
    data.drop([0, 1], inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.drop_duplicates(subset='Name', inplace=True)

    data['root'] = 'initial'
    data["plate"] = np.nan
    data["path"] = np.nan
    data["dx"] = np.zeros(data.shape[0])
    data["dy"] = np.zeros(data.shape[0])
    data[['_RAJ2000', '_DEJ2000']] = data[['_RAJ2000', '_DEJ2000']].astype(float)
    return data


parser = argparse.ArgumentParser()
parser.add_argument('--dfbs_path')
parser.add_argument('--subtypes_path')
parser.add_argument('--output_path')

args = parser.parse_args()
dfbs_path = args.dfbs_path or 'data/DFBS.csv'
subtypes_path = args.subtypes_path or 'data/DFBS_subtypes.csv'
output_path = args.output_path or 'data/Combined.csv'

df = prepareSubtypes(subtypes_path)
data = prepareData(dfbs_path)

merge_columns = ['root', '_RAJ2000', '_DEJ2000', 'Cl', 'Name', 'plate', 'path', 'dx', 'dy']
all_data = pd.concat([data[merge_columns], df[merge_columns]])
all_data.reset_index(inplace=True)
all_data.drop_duplicates(subset=['_RAJ2000', '_DEJ2000', 'Cl'], inplace=True)

all_data.to_csv(output_path, index=False)
