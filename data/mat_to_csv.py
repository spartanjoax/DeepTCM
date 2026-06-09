################################################################################
# Copyright (c) 2026 José Joaquín Peralta Abadía.                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 06-06-2026                                                             #
# Author(s): José Joaquín Peralta Abadía                                       #
# E-mail: josejoaquin.peralta.abadia@gmail.com                                 #
################################################################################

"""Convert the NASA Ames Milling Dataset from MATLAB (.mat) format to a structured CSV.

Run once before first training to produce mill_expanded.csv, which is then loaded
by datasets.py. Requires scipy.io.

Usage:
    python data/mat_to_csv.py
"""
# Original code by Vinayak Tyagi: https://www.kaggle.com/code/vinayak123tyagi/mat-to-csv-code
# Modified by José Joaquín Peralta Abadía
# Changes: Change to have one row per timepoint of the signals acquired and save resulting file to csv

import scipy.io
import pandas as pd

def convert_expanded(mat_file = 'data/mill.mat', csv_file='data/mill_expanded.csv', start_cut=2000, end_cut=7200):
    """Convert NASA Ames Milling .mat file to a structured CSV with one row per timestep.

    Args:
        mat_file:   Path to the source MATLAB file (default: ``data/mill.mat``).
        csv_file:   Destination CSV path (default: ``data/mill_expanded.csv``).
        start_cut:  First sample index to retain per run (trims sensor warm-up).
        end_cut:    Last sample index to retain per run (exclusive).
    """
    measurements = {}
    mat = scipy.io.loadmat(mat_file)

    mat = {k:v for k, v in mat.items() if k[0] != '_'}
    
    data = {}
    for k,v in mat.items():
        arr = v[0]
        for i in range(len(arr)):
            sub_arr = v[0][i]
            lst= []
            for sub_index in range(len(sub_arr)):
                if len(sub_arr[sub_index]) == 1:
                    vals = sub_arr[sub_index][0][0]
                else:
                    vals = None

                    if len(sub_arr[sub_index]) > start_cut:
                        sub_arr[sub_index] = sub_arr[sub_index][start_cut:end_cut]

                    if i in measurements.keys():
                        measurements[i][sub_index] = sub_arr[sub_index]
                    else:
                        measurements[i] = {sub_index: sub_arr[sub_index]}
                lst.append(vals)
            data[i] = lst

    counter = 0
    data_expanded = {}
    for k,v in data.items():
        if k in measurements:
            values_len = len(list(measurements[k].values())[0])

            for i in range(values_len):
                lst= []
                for sub_v in range(len(v)):
                    if v[sub_v] is not None:
                        lst.append(v[sub_v])

                for _, v2 in measurements[k].items():
                    lst.append(v2[i][0])
                data_expanded[counter] = lst
                counter += 1

    data_file = pd.DataFrame.from_dict(data_expanded, orient='index', columns=['case',
                                                                      'run',
                                                                      'VB', 
                                                                      'time',
                                                                      'DOC',
                                                                      "feed", 
                                                                      "material", 
                                                                      "smcAC",
                                                                      "smcDC",
                                                                      "vib_table",
                                                                      "vib_spindle",
                                                                      "AE_table",
                                                                      "AE_spindle"])
    
    data_file = data_file.sort_values(['case','run','time'])
    data_file.to_csv(csv_file,sep=';',decimal='.')