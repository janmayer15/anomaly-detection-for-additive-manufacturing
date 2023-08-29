'''
    IMPORTS (line 1-20)
'''
import base64
import datetime
import io
import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import json
from dash.exceptions import PreventUpdate
import os
import sys
from datetime import date
import numpy as np
import open3d as o3d
import copy
import time
import plotly.graph_objects as go
import scipy
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from tensorflow import keras
import math
import random
import dash_mantine_components as dmc
from dash_iconify import DashIconify

import graphviz
import lingam
from dash.dash_table.Format import Group
from lingam.utils import make_dot
os.environ["PATH"] += os.pathsep + r'C:\Users\Dongxu\anaconda3\Graphviz\bin/'

'''
    METHODS (line 22-1170)
'''

AXIS_VALUE  = 'argument'
AXIS_CHANGE = 'change'
class GCodeProcessor:
   retract_idx = 1

   def __init__(self):
      super().__init__()
      self.lines = []
      self.header = []
      self.footer = []
      
   def parse(self, text):
   # parse G-Code file lines and save as dictionary into : 
   #    Header
   #    Code lines : G0/G1 Commands (with or without AXIS_CHANGE), Comments 
   #    Footer

      text_array = text.split('\n')
      for i,t in enumerate(text_array):
          if t == ';START_OF_HEADER':
              header_start = i
          if t == ';END_OF_HEADER':
              header_end = i
          if t == ';End of Gcode':
              footer_start = i+1
    
      self.header = text_array[header_start+1:header_end]
      self.footer = text_array[footer_start:]
      self.lines = [{'raw': t} for t in text_array]
      last = {}
      for i in range(len(self.lines)):
         line = self.lines[i]['raw'] 
         if line.startswith('G0') or line.startswith('G1'):
            if ('Wiping+material retraction' in line or 'Compensation for the retraction' in line):
                break
            i_comment = line.find(';')
            if i_comment != -1:
               self.lines[i][';'] = line[i_comment:]
               line = line[:i_comment].strip()
               
            parts = line.split(' ')
            code = parts[0]
            self.lines[i]['code'] = code
            
            for part in parts[1:]:
               self.lines[i][part[0]] = {AXIS_CHANGE: True, AXIS_VALUE: part[1:]}
               last[part[0]] = part[1:]
            for k in ['F', 'X', 'Y', 'Z', 'E']:
               if k not in self.lines[i]:
                  self.lines[i][k] = {AXIS_CHANGE: False, AXIS_VALUE: (last[k] if k in last else '0')}
    
    
   def get_min_max_xyz(self):
       # return wall printing boundaries
       for line in self.header:
           if ';PRINT.SIZE.MIN.X' in line:
                min_x = float(line.split(':')[1])
           elif ';PRINT.SIZE.MAX.X' in line:
                max_x = float(line.split(':')[1])
           elif ';PRINT.SIZE.MIN.Y' in line:
                min_y = float(line.split(':')[1])
           elif ';PRINT.SIZE.MAX.Y' in line:
                max_y = float(line.split(':')[1])
           elif ';PRINT.SIZE.MIN.Z' in line:
                min_z = float(line.split(':')[1])
           elif ';PRINT.SIZE.MAX.Z' in line:
                max_z = float(line.split(':')[1])
       return [min_x,max_x,min_y,max_y,min_z,max_z]
           
   def _make_raw(self, line):
      if 'code' in line.keys():
         if line['E'][AXIS_CHANGE]:
            line['E'][AXIS_VALUE] = '{:.5f}'.format(float(line['E'][AXIS_VALUE]))
         arguments = ' '.join([k+line[k][AXIS_VALUE] for k in ['F', 'X', 'Y', 'Z', 'E'] if line[k][AXIS_CHANGE]])
         
         return line['code'] + ' ' + arguments + ((' '+line[';']) if ';' in line.keys() else '')
         
      return line['raw']
   
   def update_raw(self):
      for i in range(len(self.lines)):
         line = self.lines[i]
         self.lines[i]['raw'] = self._make_raw(line)

         
   
   def synthesize(self):
      return '\n'.join([l['raw'] for l in self.lines])

   def _find_next_gcode(self, i, mask=None):
   # find the g code line right after the i-th line,  and return its index
      for j in range(i, len(self.lines)):
         if mask is None or j in mask:
            keys = self.lines[j].keys()
            if 'code' in keys and self.lines[j]['code'] in ['G0', 'G1']:# and any(['X' in keys, 'Y' in keys, 'Z' in keys]):
               return j
      return None
   
   def _find_previous_gcode(self, i, mask=None):
   # find the g code right line before the i-th line, and return its index
      for j in reversed(range(0, i+1)):
         if mask is None or j in mask:
            keys = self.lines[j].keys()
            if 'code' in keys and self.lines[j]['code'] in ['G0', 'G1']:# and any(['X' in keys, 'Y' in keys, 'Z' in keys]):
               return j
      return None
   
   def _find_next_layer(self, i, mask=None):
      j = self._find_next_gcode(i+1, mask)
      z = float(self.lines[i]['Z'][AXIS_VALUE])
      while j is not None:
         if (mask is None or j in mask) and self.lines[j]['Z'][AXIS_CHANGE]:
            if float(self.lines[j]['Z'][AXIS_VALUE]) > z:
               break
         j = self._find_next_gcode(j+1, mask)
      return j

   def _indices_intersecting(self, axis, plane, mask=None):
   # loop over all G Code lines and return indices that are intersect with an input plane
      indices = []
      start = 0
      i = self._find_next_gcode(0, mask)
      while i is not None:
         end   = float(self.lines[i][axis][AXIS_VALUE])
         if (start <= plane <= end) or (start >= plane >= end):
            indices.append(i)
         i = self._find_next_gcode(i+1, mask)
         start = end
      return indices
   
   def _indices_into(self, axis, left, right, mask=None):
   # return indices of G-Codes found in the selected axis inside the selected coordinates. 
      if left > right:
         temp = right
         right = left
         left = temp
      indices = []
      i = self._find_next_gcode(0, mask)
      while i is not None:
         next = self._find_next_gcode(i+1, mask)
         pos = float(self.lines[i][axis][AXIS_VALUE])
         if left <= pos <= right and next is not None and left <= float(self.lines[next][axis][AXIS_VALUE]) <= right:
            indices.append(i)
         i = next
      return indices
   
   def _indices_between(self, axis, left, right, mask=None):
   # return indices of G-Codes found in the selected axis inside the selected coordinates, boundary included.
      if left > right:
         temp = right
         right = left
         left = temp
      indices = []
      i = self._find_next_gcode(0, mask)
      while i is not None:
         pos = float(self.lines[i][axis][AXIS_VALUE])
         if left <= pos <= right:
            indices.append(i)
         i = self._find_next_gcode(i+1, mask)
      return indices
   
   def select_between(self, bounding_box):
      box_planes = {k: [bounding_box[0][k], bounding_box[1][k]] for k in ['X', 'Y', 'Z']}
      idx = self._indices_between('X', box_planes['X'][0], box_planes['X'][1])
      idx = self._indices_between('Y', box_planes['Y'][0], box_planes['Y'][1], idx)
      idx = self._indices_between('Z', box_planes['Z'][0], box_planes['Z'][1], idx)
      return idx
   
   def select_into(self, bounding_box):
      box_planes = {k: [bounding_box[0][k], bounding_box[1][k]] for k in ['X', 'Y', 'Z']}
      idx = self._indices_into('X', box_planes['X'][0], box_planes['X'][1])
      idx = self._indices_into('Y', box_planes['Y'][0], box_planes['Y'][1], idx)
      idx = self._indices_into('Z', box_planes['Z'][0], box_planes['Z'][1], idx)
      return idx
   
   def _delta(self, gcode_index):
      return self._print_vector(gcode_index) - self._print_vector(self._find_previous_gcode(gcode_index-1))
      
   def _print_vector(self, gcode_index):
      if gcode_index is None:
         return np.zeros(4)
      return np.array([float(self.lines[gcode_index][k][AXIS_VALUE]) for k in ['X', 'Y', 'Z', 'E']])

   def _start_vector(self, gcode_index):
       return self._vector(self._find_previous_gcode(gcode_index-1))   
   
   def _vector(self, gcode_index):
      if gcode_index is None:
         return np.zeros(4)
      return np.array([float(self.lines[gcode_index][k][AXIS_VALUE]) for k in ['X', 'Y', 'Z', 'E']])
  
   def add_vector(self, gcode_index, vector):
      self.lines[gcode_index]['X'][AXIS_VALUE] = '{:.2f}'.format(float(self.lines[gcode_index]['X'][AXIS_VALUE]) + vector[0])
      self.lines[gcode_index]['Y'][AXIS_VALUE] = '{:.2f}'.format(float(self.lines[gcode_index]['Y'][AXIS_VALUE]) + vector[1])
      self.lines[gcode_index]['Z'][AXIS_VALUE] = '{:.2f}'.format(float(self.lines[gcode_index]['Z'][AXIS_VALUE]) + vector[2])
      self.lines[gcode_index]['E'][AXIS_VALUE] = '{:.5f}'.format(float(self.lines[gcode_index]['E'][AXIS_VALUE]) + vector[3])
   
   def propagate_add_vector(self, gcode_index, vector):
      i = self._find_next_gcode(gcode_index)
      while i is not None:
         self.add_vector(i, vector)
         i = self._find_next_gcode(i+1)
   
   def set_extrusion(self, gcode_index, delta_e):
      i = self._find_next_gcode(gcode_index, None)
      j = self._find_next_gcode(i+1, None)
      if i is not None and j is not None:
         add_e = (float(self.lines[i]['E'][AXIS_VALUE]) + delta_e) - float(self.lines[j]['E'][AXIS_VALUE])
         while j is not None:   # Propagate
            previous_e = float(self.lines[j]['E'][AXIS_VALUE])
            self.lines[j]['E'][AXIS_VALUE] = str(previous_e + add_e)
            j = self._find_next_gcode(j+1, None)
   
   def set_extrusion_scale(self, gcode_index, factor, extrusion_type = 1):
      # overextrusion: factor = 1 -> 100 % + factor*100% = 200% material inside B-Box
      # underextrusion: factor = 0.3 -> 100% - factor*100% = 70% material inside B-Box
      i = self._find_next_gcode(gcode_index, None)
      j = self._find_next_gcode(i+1, None)
      if i is not None and j is not None:
         add_e = (float(self.lines[j]['E'][AXIS_VALUE]) - float(self.lines[i]['E'][AXIS_VALUE])) * factor
         #print('add_e', add_e)
         while j is not None:   # Propagate
            previous_e = float(self.lines[j]['E'][AXIS_VALUE])
            self.lines[j]['E'][AXIS_VALUE] = str(previous_e + add_e * extrusion_type)
            j = self._find_next_gcode(j+1, None)

   def _split_movement(self, gcode_index, cut_plane):
      # plane: 4d-array, first 3 are the normal vector, last one is bias.
      start = self._start_vector(gcode_index)
      end   = self._vector(gcode_index)
      g_start = self.lines[gcode_index]
      plane_n = np.array(cut_plane[0:3])
      plane_b = cut_plane[3]
      
      move_vector = end - start
      delta_e = move_vector[3]
      move_vector = move_vector[:3]
      start_e = start[3]
      end_e = end[3]
      start = start[:3]
      end = end[:3]
      # Compute intersection move_vector with plane.
      denominator = np.dot(move_vector, plane_n)
      if denominator == 0.:
         # move_vector and plane are in parallel --> no intersection.
         print('NO INTERSECTION AT GCODE:', g_start)
         return None
      else:
         c = (plane_b - np.dot(start, plane_n)) / denominator
      breakpoint = move_vector*c + start
      gcode_new = copy.deepcopy(g_start)
      if g_start['X']['change']:
         gcode_new['X'][AXIS_VALUE] = str(breakpoint[0])
      if g_start['Y']['change']:
         gcode_new['Y'][AXIS_VALUE] = str(breakpoint[1])
      if g_start['Z']['change']:
         gcode_new['Z'][AXIS_VALUE] = str(breakpoint[2])
      if g_start['E']['change']:
         gcode_new['E'] = {'change': True, AXIS_VALUE: str(start_e + c*delta_e)}
      
      gcode_new['raw'] = self._make_raw(gcode_new)
      # gcode_new is the code from the original start to the breakpoint.

      return gcode_new
   
   def set_extrusion_factor(self, bounding_box_corners, factor):
      # Select GCodes that intersect with bounding_box. bounding_box is axis-parallel.
      box_planes = {k: [bounding_box_corners[0][k], bounding_box_corners[1][k]] for k in ['X', 'Y', 'Z']}
      plane_y0 = box_planes['Y'][0]
      plane_y1 = box_planes['Y'][1]
      between_z = self._indices_between('Z', box_planes['Z'][0], box_planes['Z'][1])
      rest = self._indices_between('X', box_planes['X'][0], box_planes['X'][1], between_z)
      
      intersecting = self._indices_intersecting('Y', plane_y0, rest)
      inserts = []
      within_indices = []
      for i in reversed(intersecting):
         pos = float(self.lines[self._find_next_gcode(i+1)]['Y'][AXIS_VALUE])
         p_y0 = plane_y0
         p_y1 = plane_y1
         if pos < plane_y0:
            # pos is left of plane_y0, which is left to plane_y1
            p_y0 = plane_y1
            p_y1 = plane_y0
         gcode_new1 = self._split_movement(i, [0, 1, 0, p_y0])
         gcode_new2 = self._split_movement(i, [0, 1, 0, p_y1])
         
         if gcode_new1 is not None:
            inserts.append((i, gcode_new2))
            inserts.append((i, gcode_new1))   # gcode_new1 is before gcode_new2
            within_indices.append(i)
      for v in inserts:
         self.lines.insert(v[0], v[1])

      return within_indices

   def add_retract(self, under_pos):
      # Add material retraction when transtioniong from/to underextrusion to prevent material overdraw outside bounding box
      shift = len(under_pos)*2
  
      idx_0 = 0
      idx_1 = 1
      count = 0
      while idx_0 < len(under_pos) and idx_1 < len(under_pos):
         under_pos[idx_0] = under_pos[idx_0] - 3 + shift - (count * 4)
         under_pos[idx_1] = under_pos[idx_1] - 4 + shift - (count * 4)
         idx_0 = idx_0 + 2
         idx_1 = idx_1 + 2
         count = count + 1
      
      for i in under_pos:
         self.lines[i]['raw'] =  self.lines[i]['raw'] + "\nM83 ;relative extrusion mode\nG0 E-2.75 ;material retraction\nM82 ;absolute extrusion mode"
       
def do_overextrusion(bbox, proc, amount=None, factor=None):
       # Split code -----a----->
       #       into -b->-c->-d->
       # then set
       #      b[E] += increment
       #      propagate_increment()
       proc.set_extrusion_factor(bbox, 1)
       idx = proc.select_into(bbox)
       for i in reversed(idx):
          if amount is not None:
             proc.set_extrusion(i, amount)
          else:
             proc.set_extrusion_scale(i, factor)
          
       proc.update_raw()
       revlist = list(reversed(idx))
         
def do_underextrusion(bbox, proc, amount=None, factor=None):
       # Split code -----a----->
       #       into -b->-c->-d->
       # then set
       #      b[E] += increment
       #      propagate_increment()
       under_pos = proc.set_extrusion_factor(bbox, 1)
       idx = proc.select_into(bbox)
       for i in reversed(idx):
          if amount is not None:
             proc.set_extrusion(i, amount)
          else:
             proc.set_extrusion_scale(i, factor, extrusion_type = -1)
          
       proc.update_raw()
       #proc.add_retract(under_pos)
       revlist = list(reversed(idx))

def delBorder(data: np.array, side: str, originalBorders: list):
    '''
    delBorder finds the koordinate to which all close-meshed structures at
    given side reach and deletes all points up to the found koordinate
    
    _________________________params/returns__________________________________
                                                 
                                                     [[x1, x2, x3, ...], 
    :param data: raw scan as (3, n) shaped np.array:  [y1, y2, y3, ...], 
                                                      [z1, z2, z3, ...]]                                     
    :param side: inspected side (X_min, X_max, Y_min or Y_max)
    :param originalBorders: list with raw scan min/max values [X_min, X_max, Y_min, Y_max]

    :return: filtered df
    '''
    print(len(data[0]))
    data_del = np.array([0,0,0])
    n_subs = 20
    if side == "X_min":
        prev = None
        subParts = (np.max(data[1]) - np.min(data[1]))/n_subs
        for j in range(0, n_subs):
            data_j = np.take(data, np.where(data[1] >= np.min(data[1]) + subParts*j)[0], axis=1)
            data_j = np.take(data_j, np.where(data_j[1] < np.min(data[1]) + subParts*(j+1))[0], axis=1)

            x_points = np.unique(np.sort(data_j[0])) # different x points in data
            step_size = np.min(np.unique(np.sort(data[0]))[1:] - np.unique(np.sort(data[0]))[:-1]) # scanner resolution

            xMin = x_points[0] # initialization of the filter koordinate
            if xMin < originalBorders[0] + 10:
                # moves the filter coordinate away from the border as long as the structure is connected
                for i in range(1, len(x_points)-1):
                    if x_points[i] - x_points[i-1] < step_size*2: # connected structure if stepsize is kept
                        xMin = x_points[i-1]
                    else:
                        break
            
            if xMin > originalBorders[0] + 15:
                xMin = originalBorders[0] + 15
            if prev == None:
                prev = xMin
            elif xMin > prev:
                xMin = prev
            prev = xMin
            data_j = np.take(data_j, np.where(data_j[0] > xMin)[0], axis=1)
            data_del = np.vstack((data_del, data_j.T))
        print(len(data_del))
        return data_del[1:].T

    if side == "X_max":
        prev = None
        subParts = (np.max(data[1]) - np.min(data[1]))/n_subs
        for j in range(0, n_subs):
            data_j = np.take(data, np.where(data[1] >= np.min(data[1]) + subParts*j)[0], axis=1)
            data_j = np.take(data_j, np.where(data_j[1] < np.min(data[1]) + subParts*(j+1))[0], axis=1)
        
            x_points = np.unique(np.sort(data_j[0])) # different x points in data
            step_size = np.min(np.unique(np.sort(data[0]))[1:] - np.unique(np.sort(data[0]))[:-1]) # scanner resolution

            xMax = x_points[-1] # initialization of the filter koordinate
            if xMax > originalBorders[1] - 10:
                # moves the filter coordinate away from the border as long as the structure is connected
                for i in reversed(range(1, len(x_points)-1)):
                    if x_points[i] - x_points[i-1] < step_size*2: # connected structure if stepsize is kept
                        xMax = x_points[i-1]
                    else:
                        break
            
            if xMax < originalBorders[1] - 15:
                xMax = originalBorders[1] - 15
            if prev == None:
                prev = xMax
            elif xMax < prev:
                xMax = prev
            prev = xMax
            data_j = np.take(data_j, np.where(data_j[0] < xMax)[0], axis=1)
            data_del = np.vstack((data_del, data_j.T))
        print(len(data_del))
        return data_del[1:].T

    if side == "Y_min":
        prev = None
        subParts = (np.max(data[0]) - np.min(data[0]))/n_subs
        for j in range(0, n_subs):
            data_j = np.take(data, np.where(data[0] >= np.min(data[0]) + subParts*j)[0], axis=1)
            data_j = np.take(data_j, np.where(data_j[0] < np.min(data[0]) + subParts*(j+1))[0], axis=1)

            y_points = np.unique(np.sort(data_j[1])) # different y points in data
            step_size = np.min(np.unique(np.sort(data[1]))[1:] - np.unique(np.sort(data[1]))[:-1]) # scanner resolution

            yMin = y_points[0] # initialization of the filter koordinate
            if yMin < originalBorders[2] + 10:
                # moves the filter coordinate away from the border as long as the structure is connected
                for i in range(1, len(y_points)-1):
                    if y_points[i] - y_points[i-1] < step_size*2: # connected structure if stepsize is kept
                        yMin = y_points[i-1]
                    else:
                        break
            
            if yMin > originalBorders[2] + 15:
                yMin = originalBorders[2] + 15
            if prev == None:
                prev = yMin
            elif yMin > prev:
                yMin = prev
            prev = yMin
            data_j = np.take(data_j, np.where(data_j[1] > yMin)[0], axis=1)
            data_del = np.vstack((data_del, data_j.T))
        print(len(data_del))
        return data_del[1:].T

    if side == "Y_max":
        prev = None
        subParts = (np.max(data[0]) - np.min(data[0]))/n_subs
        for j in range(0, n_subs):
            data_j = np.take(data, np.where(data[0] >= np.min(data[0]) + subParts*j)[0], axis=1)
            data_j = np.take(data_j, np.where(data_j[0] < np.min(data[0]) + subParts*(j+1))[0], axis=1)

            y_points = np.unique(np.sort(data_j[1])) # different y points in data
            step_size = np.min(np.unique(np.sort(data[1]))[1:] - np.unique(np.sort(data[1]))[:-1]) # scanner resolution

            yMax = y_points[-1] # initialization of the filter koordinate
            if yMax > originalBorders[3] - 10:
                # moves the filter coordinate away from the border as long as the structure is connected
                for i in reversed(range(1, len(y_points)-1)):
                    if y_points[i] - y_points[i-1] < step_size*2: # connected structure if stepsize is kept
                        yMax = y_points[i-1]
                    else:
                        #print((len(y_points)-1, i))
                        break
            #print("yMax", yMax)
            if yMax < originalBorders[3] - 15:
                yMax = originalBorders[3] - 15
            #if prev == None:
            #    prev = yMax
            #elif yMax < prev:
            #    yMax = prev
            prev = yMax
            data_j = np.take(data_j, np.where(data_j[1] < yMax)[0], axis=1)
            data_del = np.vstack((data_del, data_j.T))
    
        return data_del[1:].T

def delAllBorders(df: pd.DataFrame):
    '''
    delAllBorders deltes all close-meshed structures at the sides of the raw
    scan
    
    _________________________params/returns__________________________________
                                                   
    :param df: raw scan with all parts (n, 3) shaped pandas.DataFrame 
               (column names: "X", "Y", "Z")                                     

    :return: filtered df
    '''

    xSpan = np.abs(np.max(df["X"]) - np.min(df["X"])) # maximum range in x direction
    ySpan = np.abs(np.max(df["Y"]) - np.min(df["Y"])) # maximum range in y direction

    # calculates a DataFrame summarizing the number of points in 5% range at each side
    order_df = pd.DataFrame({"X_min": [len(df[df["X"] < (np.min(df["X"]) + xSpan/30)])],
                             "X_max": [len(df[df["X"] > (np.max(df["X"]) - xSpan/30)])],
                             "Y_min": [len(df[df["Y"] < (np.min(df["Y"]) + ySpan/30)])],
                             "Y_max": [len(df[df["Y"] > (np.max(df["Y"]) - ySpan/30)])]})
    
    # deletes the unwanted structure at each side in the order from the side with most points to less points
    data = df.to_numpy().T
    borders, i = [np.min(data[0]), np.max(data[0]), np.min(data[1]), np.max(data[1])], 0
    for c in order_df.sort_values(by=0, axis=1, ascending=False).columns:
        if i >= 2:
            break        
        data = delBorder(data, c, borders)
        i += 1

    print("last", len(data[0]))
    return data

def findPart(data: np.array, x: float, y: float, step_size: float):
    '''
    findPart locates the part where the given point (x and y) is in and
    returns its rough dimensions
    
    _________________________params/returns__________________________________
                                                
                                                         [[x1, x2, x3, ...], 
    :param data: overall scan as (3, n) shaped np.array:  [y1, y2, y3, ...], 
                                                          [z1, z2, z3, ...]]                               
    :param x: given points x koordinate
    :param y: given points y koordinate 
    :param step_size: scanner resolution
                                                          
    :return: rough part dimensions in the form of [xMin, xMax, yMin, yMax]
    '''
    
    # Step 1: find lower left corner

    # filter given data to only contain points lower and more left than the given point
    data_filtered = np.take(data, np.where(data[0] < x)[0], axis=1)
    data_filtered = np.take(data_filtered, np.where(data_filtered[1] < y)[0], axis=1)

    x_points_filtered = np.unique(np.sort(data_filtered[0])) # different x points in filtered data    
    
    # find x koordinate to which the part is a close-meshed structure
    for i in reversed(range(len(x_points_filtered))):
        if i == 0: # found corners x koordinate if no points left
            xMin = x_points_filtered[0]
            break
        if x_points_filtered[i] - x_points_filtered[i-1] > step_size*2.1: # found corners x koordinate if points arent close-meshed anymore
            xMin = x_points_filtered[i]
            break

    y_points_filtered = np.unique(np.sort(data_filtered[1])) # different y points in filtered data  

    # find y koordinate to which the part is a close-meshed structure
    for i in reversed(range(len(y_points_filtered))):
        if i == 0: # found corners y koordinate if no points left
            yMin = y_points_filtered[0]
            break
        if y_points_filtered[i] - y_points_filtered[i-1] > step_size*2.1: # found corners y koordinate if points arent close-meshed anymore
            yMin = y_points_filtered[i]
            break

    # Step 2: find upper right corner

    # filter given data to only contain points upper and more right than the given point up to a defined maximum x and y
    data_filtered = np.take(data, np.where(data[0] > x)[0], axis=1)
    data_filtered = np.take(data_filtered, np.where(data_filtered[1] > y)[0], axis=1)
    data_filtered = np.take(data_filtered, np.where(data_filtered[0] < x+24)[0], axis=1)
    data_filtered = np.take(data_filtered, np.where(data_filtered[1] < y+24)[0], axis=1)

    x_points_filtered = np.unique(np.sort(data_filtered[0])) # different x points in filtered data

    # find x koordinate to which the part is a close-meshed structure
    for i in range(len(x_points_filtered)):
        if i == len(x_points_filtered)-1: # found corners x koordinate if no points left
            xMax = x_points_filtered[-1]
            break
        if x_points_filtered[i+1] - x_points_filtered[i] > step_size*2.1: # found corners x koordinate if points arent close-meshed anymore
            xMax = x_points_filtered[i]
            break

    y_points_filtered = np.unique(np.sort(data_filtered[1])) # different y points in filtered data 

    # find y koordinate to which the part is a close-meshed structure
    for i in range(len(y_points_filtered)):
        if i == len(y_points_filtered)-1: # found corners y koordinate if no points left
            yMax = y_points_filtered[-1]
            break
        if y_points_filtered[i+1] - y_points_filtered[i] > step_size*2.1: # found corners y koordinate if points arent close-meshed anymore
            yMax = y_points_filtered[i]
            break

    # Step 3: correct for slight error as cause of a wrong part angle in the scan

    # correct smaller part direction to ensure a quadratic shape
    if (xMax - xMin) > (yMax - yMin):
        correctionFactor = ((xMax - xMin) - (yMax - yMin))/2
        yMin = yMin - correctionFactor
        yMax = yMax + correctionFactor
    elif (xMax - xMin) < (yMax - yMin):
        correctionFactor = ((yMax - yMin) - (xMax - xMin))/2
        xMin = xMin - correctionFactor
        xMax = xMax + correctionFactor

    xMin, xMax, yMin, yMax = xMin-1, xMax+1, yMin-1, yMax+1 # add safety distance on each side

    return [xMin, xMax, yMin, yMax]

def findAllParts(data: np.array):
    '''
    findAllParts locates all parts in the given scan (without side structures) 
    and returns their rough dimensions
    
    ____________________part order in returned list__________________________
                         _______________
        hole scan ->    | :             |
                        | [7] [8] [9]   |
        parts -> [X]    | [4] [5] [6]   |
                        | [1] [2] [3].. |
        order -> X      |_______________|

    _________________________params/returns__________________________________

                                                         [[x1, x2, x3, ...], 
    :param data: raw scan as (3, n) shaped np.array:      [y1, y2, y3, ...], 
                                                          [z1, z2, z3, ...]]                                                          
    :return: list with the rough part dimensions for all parts
    '''

    agg_size = 8
    step_size = np.min(np.unique(np.sort(data[0]))[1:] - np.unique(np.sort(data[0]))[:-1]) # scanner resolution
    fullPointDens = agg_size/step_size * agg_size/step_size # estimated number of points in a 1 by 1 quare if fully closed meshed

    #x_points = np.unique(np.sort(data.round(decimals=0)[0])) # integer points in x direction
    #y_points = np.unique(np.sort(data.round(decimals=0)[1])) # integer points in y direction
    x_points = np.arange(np.min(data[0]), np.max(data[0]), agg_size)
    y_points = np.arange(np.min(data[1]), np.max(data[1]), agg_size)

    partsDims = [] # initialization of the return list

    # iterate over all aggregated points
    for y in y_points[1:]:
        for x in x_points[1:]:

            xFilter = data[1][np.where((data[0] >= x-agg_size) & (data[0] <= x))] # data points between current and last x
            # current point (x and y) lays in a part if the sqare x-1 to x and y-1 to y has a near to full point density
            if len(np.where((xFilter >= y-agg_size) & (xFilter <= y))[0]) > 100:
                print((len(np.where((xFilter >= y-agg_size) & (xFilter <= y))[0]), fullPointDens*0.30, x-agg_size, x, y-agg_size, y))
            if len(np.where((xFilter >= y-agg_size) & (xFilter <= y))[0]) > fullPointDens*0.30:
                print("points", (x,y))
                partsDims.append(findPart(data, x, y, step_size)) # find part dimension
                
                # delete points in found part dimension for no further consideration
                xVals = np.where(data[0] < partsDims[-1][1])
                yVals = np.where(data[1] < partsDims[-1][3])
                data = np.delete(data, np.intersect1d(xVals, yVals), axis=1)
            else:
                xFilter = data[1][np.where((data[0] >= x) & (data[0] <= x+agg_size))] # data points between current and last x
                if len(np.where((xFilter >= y) & (xFilter <= y+agg_size))[0]) > 100:
                    print((len(np.where((xFilter >= y) & (xFilter <= y+agg_size))[0]), fullPointDens*0.30, x, x+agg_size, y, y+agg_size))
                if len(np.where((xFilter >= y) & (xFilter <= y+agg_size))[0]) > fullPointDens*0.30:
                    print("points", (x+agg_size,y+agg_size))
                    partsDims.append(findPart(data, x+agg_size, y+agg_size, step_size)) # find part dimension
                
                    # delete points in found part dimension for no further consideration
                    xVals = np.where(data[0] < partsDims[-1][1])
                    yVals = np.where(data[1] < partsDims[-1][3])
                    data = np.delete(data, np.intersect1d(xVals, yVals), axis=1)

    return partsDims

def indexParts(dims: list, real_i: list):
    dims_df = pd.DataFrame({"xMin": [dims[0][0]], "xMax": [dims[0][1]], "yMin": [dims[0][2]], "yMax": [dims[0][3]]})
    for i in range(1, len(dims)):
        dims_df = pd.concat([dims_df, pd.DataFrame({"xMin": [dims[i][0]], "xMax": [dims[i][1]], "yMin": [dims[i][2]], "yMax": [dims[i][3]]})], ignore_index=True)
    
    dims_df.sort_values(by=["yMin"], inplace=True, ignore_index=True)
    rows, row_i = [], 0
    for index, row in dims_df.iterrows():
        if index == 0:
            rows.append(row_i)
        elif row["yMin"] < dims_df.loc[index-1, "yMin"] + 10:
            rows.append(row_i)
        else:
            row_i += 1
            rows.append(row_i)    
    dims_df["row"] = rows

    dims_df.sort_values(by=["xMin"], inplace=True, ignore_index=True)
    cols, col_i = [], 0
    for index, row in dims_df.iterrows():
        if index == 0:
            cols.append(col_i)
        elif row["xMin"] < dims_df.loc[index-1, "xMin"] + 10:
            cols.append(col_i)
        else:
            col_i += 1
            cols.append(col_i)    
    dims_df["col"] = cols

    dims_df.sort_values(by=["col"], inplace=True, ignore_index=True, ascending=True)
    dims_df.sort_values(by=["row"], inplace=True, ignore_index=True, ascending=False)

    dims_df["real_i"] = real_i
    dims_df.sort_values(by=["real_i"], inplace=True, ignore_index=True, ascending=True)

    return dims_df.to_numpy()[:,:-1]

def to_ply(df: pd.DataFrame, path: str, filename: str):
    '''
    to_ply converts the given pandas DataFrame to a .ply file and saves it to
    the given directory with the given name
    
    _________________________params/returns__________________________________
                                                                             
    :param df: (n, m) shaped pandas.DataFrame
    :param path: path to destinated directory
    :param filename: intended name of file
                                                          
    :return: None
    '''
    
    if type(df) == pd.DataFrame:
        # create ply header
        ply_text = 'ply\r\nformat ascii 1.0\r\nelement vertex ' + str(len(df)-1) + '\r\nproperty float x\r\nproperty float y\r\nproperty float z\r\nproperty uchar red\r\nproperty uchar green\r\nproperty uchar blue\r\nend_header\r\n'

        df["ply"] = ["128 128 128"] * len(df) # add new column to dataframe

        # wirte header and df to .ply file
        with open(path + filename, 'w') as f:
            f.write(ply_text)
            dfAsString = df.to_string(header=False, index=False)
            f.write(dfAsString)
            f.close()
    else:
        raise ValueError('Cannot store this format to ply')

def read_ply(path: str, filename: str):
    '''
    read_ply converts the given pandas DataFrame to a .ply file and saves it to
    the given directory with the given name
    
    _________________________params/returns__________________________________
                                                                             
    :param path: path to file directory
    :param filename: name of file with or without file extension
                                                          
    :return: (n, 3) shaped pandas.DataFrame (column names: "X", "Y", "Z")
    '''
    
    # check for correct file extension
    if filename[-4:] != ".ply":
        filename = filename + ".ply"

    # open file and extract single lines 
    file = open(path + filename, encoding="utf8")
    text = file.read()
    lines = text.split('\n')
    assert lines[0].strip() == 'ply'
    
    points, end_header = [], False # Initialize data list and header variable 

    # Iterate over each line
    for i in range(len(lines)):
        
        # extract data points
        if end_header:
            data = lines[i].strip().split(' ')
            if len(data) > 1:
                for i in range(3):
                    if '' in data:
                        data.remove('')
                    if ' ' in data:
                        data.remove(' ')
                    if "128" in data:
                        data.remove("128")
                points.append([float(d) for d in data])
        
        # check if still in header
        else:
            end_header = lines[i].strip() == 'end_header'

    return pd.DataFrame([{'X': d[0], 'Y': d[1], 'Z': d[2]} for d in points])

def correctMissingPoints(data_x_val: np.array, y_vals: np.array, data: np.array, stepsize: float):
    '''
    correctMissingPoints detects gaps in a given row (fixed x value) due to 
    measurement errors in the scan and fills them with the average z-value of 
    the surrounding points. This is a recursive function!
    
    _________________________params/returns__________________________________
                                                                             
    :param df_x_val: row from scanner DataFrame with fixed x value
    :param y_vals: y values to be looked at for gaps
    :param data: complete scan file as np.array (3, n) shaped
    :param stepsize: scanner resolution
                                                          
    :return: colplete scan with corrected row as (3, n) shaped np.array
    '''

    # calculate row at the given y values
    y_col_data = np.take(data_x_val, np.where(data_x_val[1] >= y_vals[0])[0], axis=1)
    y_col_data = np.take(y_col_data, np.where(y_col_data[1] <= y_vals[-1])[0], axis=1)
    y_col_data = y_col_data[:, y_col_data[1,].argsort()]
    
    # no points to be looked at
    if len(y_col_data[0]) == 0:
        return data
    
    # row is fully meshed with the given stepsize
    if (len(y_col_data[0]) - 1) * stepsize == y_col_data[1][-1] - y_col_data[1][0]:
        return data

    # iterate over points if few enought
    else:
        for i in range(len(y_vals)):
            # point missing and point not on the edge of the scan
            if len(np.where(y_col_data[1] == y_vals[i])[0]) == 0 and not len(np.where(data_x_val[1] < y_vals[i])[0]) == 0 and not len(np.where(data_x_val[1] > y_vals[i])[0]) == 0 and not len(np.where(data[0] > data_x_val[0][0])[0]) == 0 and not len(np.where(data[0] < data_x_val[0][0])[0]) == 0:
                
                # calculate value to fill gap with
                fitval = np.take(data, np.where(data[0] >= data_x_val[0][0]-0.05)[0], axis=1)
                fitval = np.take(fitval, np.where(fitval[0] <= data_x_val[0][0]+0.05)[0], axis=1)
                fitval = np.take(fitval, np.where(fitval[1] >= y_vals[i]-0.05)[0], axis=1)
                fitval = np.take(fitval, np.where(fitval[1] <= y_vals[i]+0.05)[0], axis=1)
                fitval = np.mean(fitval[2])
                
                # fill gap if possible
                if np.isnan(fitval) == False:
                    data = np.append(data, np.array([[data_x_val[0][0]], [y_vals[i]], [fitval]]), axis=1)
        return data

def rotatePart(df: pd.DataFrame, path: str, name: str):
    # convert df to .ply
    df.reset_index(drop=True, inplace=True)
    to_ply(copy.deepcopy(df), path, name + ".ply")

    # rotate .ply with open3d library with calculated negative part angele
    pcd = o3d.io.read_point_cloud(path + name + ".ply")
    R = pcd.get_rotation_matrix_from_xyz((0, 0, -np.pi/2)) # np.arctan(-m)
    pcd = copy.deepcopy(pcd.rotate(R, (0.5,0.5,0)))
    o3d.io.write_point_cloud(path + name + ".ply", pcd, write_ascii=True)

    # convert back to DataFrame
    df = read_ply(path, name + ".ply")
    os.remove(path + name + ".ply")
    df.to_csv(path + name + ".csv", index=False)

    return df

def clearPart(df: pd.DataFrame, path: str, name: str):
    '''
    clearParts clears the part from:
        - outliers outside the real part dimensions
        - missing points in the scan due to measurement errors
        - a not staight rotation
        - uneven points at the edges
        - applys standardized dimensions (0 to 1 in x and y direction)
        - saves the scan as .csv at the given path with the given name
    
    _________________________params/returns__________________________________
                                                                             
    :param df: colplete scan as (n, 3) shaped pandas.DataFrame 
               (column names: "X", "Y", "Z")
    :param path: path to destinated directory
    :param name: intended name of file
                                                          
    :return: None
    '''

    # Step 1: deltete outliers outside the real part dimensions

    df_ZOutliers_plus = df[df["Z"] > (np.mean(df["Z"]) + 3*np.std(df["Z"]))] # serach for z value outliers
    df_ZOutliers_minus = df[df["Z"] < (np.mean(df["Z"]) - 3*np.std(df["Z"]))]
    df_ZOutliers = pd.concat([df_ZOutliers_plus, df_ZOutliers_minus])

    max_X, min_X, max_Y, min_Y = np.max(df["X"]), np.min(df["X"]), np.max(df["Y"]), np.min(df["X"])
    
    # checks for all z outliers, if at the edges of the scan
    for index, row in df_ZOutliers.iterrows():
        if row["X"] > max_X-3 or row["X"] < min_X+3 or row["Y"] > max_Y-3 or row["X"] < min_Y+3:
            df.drop([index], inplace=True) # delte if at edge

    print("(1/5): deleted outlier points.")
    # Step 1: done!


    # Step 2: correct missing points in the scan due to measurement errors
    print("(2/5): adding missing points ...")

    stepsize = sorted(set(list(df["X"])))[50] - sorted(set(list(df["X"])))[49] # scanner resolution
    val_x = sorted(set(list(df["X"])))
    val_x_max = max(val_x)
    val_x_min = min(val_x)

    data = df.to_numpy().T

    # for each column with fixed x value apply method correctMissingPoints
    for x in val_x:
        sys.stdout.write("\r[" + "=" * int(((x - val_x_min) / (val_x_max - val_x_min)) * 20) + " " * int(20 - ((x - val_x_min) / (val_x_max - val_x_min)) * 20) + "] " + str(int(((x - val_x_min) / (val_x_max - val_x_min)) * 100)) + "%")
        sys.stdout.flush()
        data_val_x = np.take(data, np.where(data[0] == x)[0], axis=1)
        data = correctMissingPoints(data_val_x, np.unique(np.sort(data[1])), data, stepsize)

    df = pd.DataFrame({"X": data[0], "Y": data[1], "Z": data[2]})

    print("(2/5): done!")
    # Step 2: done!


    # Step 3: correct a not staight rotation

    # save listes with outer x and y values 
    maxs, mins = {"X": [], "Y": []}, {"X": [], "Y": []}
    for val_x in sorted(set(list(df["X"])))[50:-50]:
        maxs["X"].append(val_x)
        maxs["Y"].append(max(df[df["X"] == val_x]["Y"]))
        mins["X"].append(val_x)
        mins["Y"].append(min(df[df["X"] == val_x]["Y"]))

    # calculate part angle through linear regression at the edges
    m_max, b1_max = np.polyfit(maxs["X"], maxs["Y"], 1)
    m_min, b1_min = np.polyfit(mins["X"], mins["Y"], 1)
    m = (m_max + m_min) / 2 # average gradient
    center = ((max(df["X"]) - min(df["X"])) / 2, (max(df["Y"]) - min(df["Y"])) / 2, 0)
    
    # convert df to .ply
    df.reset_index(drop=True, inplace=True)
    to_ply(copy.deepcopy(df), path, name + ".ply")

    # rotate .ply with open3d library with calculated negative part angele
    pcd = o3d.io.read_point_cloud(path + name + ".ply")
    R = pcd.get_rotation_matrix_from_xyz((0, 0, np.arctan(-m)))
    pcd = copy.deepcopy(pcd.rotate(R, center))
    o3d.io.write_point_cloud(path + name + ".ply", pcd, write_ascii=True)

    # convert back to DataFrame
    df = read_ply(path, name + ".ply")
    os.remove(path + name + ".ply")

    print("(3/5): adjusted orientation")
    # Step 3: done!

    # Step 4: delete uneven points at the edges
    print("(4/5): calculating even edges ...")

    # x sides
    # aggregate points by rounding the x values
    df_x = copy.deepcopy(df)
    df_x["X"] = df_x["X"].round(1)

    # calculate average points per fixed x value
    mean_val_count = len(df_x[df_x["X"] == df_x.loc[0, "X"]])
    for val in set(list(df_x["X"])):
        mean_val_count = (mean_val_count + len(df_x[df_x["X"] == val])) / 2

    # find correct x borders
    min_X, max_X, min_finished = min(df_x["X"]), max(df_x["X"]), False
    for val in sorted(set(list(df_x["X"]))):
        if min_finished == False:
            if len(df_x[df_x["X"] == val]) < mean_val_count * 0.99:
                min_X = val
            else:
                min_finished = True
        elif len(df_x[df_x["X"] == val]) < mean_val_count * 0.99 and val < max_X:
            max_X = val
        elif len(df_x[df_x["X"] == val]) >= mean_val_count * 0.99:
            max_X = max(df_x["X"])

    # apply correct x dimensions
    df = df[df["X"] > min_X]
    df = df[df["X"] < max_X]

    # y sides
    # aggregate points by rounding the y values
    df_y = copy.deepcopy(df)
    df_y["Y"] = df_y["Y"].round(1)

    # calculate average points per fixed x value
    mean_val_count = len(df_y[df_y["Y"] == list(df_y["Y"])[0]])
    for val in set(list(df_y["Y"])):
        mean_val_count = (mean_val_count + len(df_y[df_y["Y"] == val])) / 2

    # find correct x borders
    min_Y, max_Y, min_finished = min(df_y["Y"]), max(df_y["Y"]), False
    for val in sorted(set(list(df_y["Y"]))):
        if min_finished == False:
            if len(df_y[df_y["Y"] == val]) < mean_val_count * 0.99:
                min_Y = val
            else:
                min_finished = True
        elif len(df_y[df_y["Y"] == val]) < mean_val_count * 0.99 and val < max_Y:
            max_Y = val
        elif len(df_y[df_y["Y"] == val]) >= mean_val_count * 0.99:
            max_Y = max(df_y["Y"])

    # apply correct x dimensions
    df = df[df["Y"] > min_Y]
    df = df[df["Y"] < max_Y]

    print("(4/5): done!")
    # Step 4: done!

    # Step 5: apply standardized dimensions

    df["X"] = df["X"] - min(df["X"])
    df["X"] = df["X"] / max(df["X"])
    df["Y"] = df["Y"] - min(df["Y"])
    df["Y"] = df["Y"] / max(df["Y"])

    df.sort_values(by=["X", "Y"], inplace=True, ignore_index=True)

    print("(5/5): standardized dimensions")
    # Step 5: done!

    return df

def downScaling(data: np.array, stepsize: float=0.02, returnType: str="array"):
    '''
    downScaling aggregates the cleaned scanner data

    _________________________params/returns__________________________________

                                                         [[x1, x2, x3, ...], 
    :param data: cleaned scan as (3, n) shaped np.array:  [y1, y2, y3, ...], 
                                                          [z1, z2, z3, ...]]
    :param stepsize: aggregation size

    :return: aggregated scanner data with aggregated z-values by mean in matrix form
    '''
    
    data_agg = np.array([[0, 0, 0]]) # initialization of the return array
    data_matrix = np.array([])
    
    stepsX = np.arange(np.min(data[0]), np.max(data[0])+stepsize, stepsize) # aggregation steps in X direction
    stepsY = np.arange(np.min(data[1]), np.max(data[1])+stepsize, stepsize) # aggregation steps in Y direction

    # iteration over each aggregation point
    for x in stepsX[1:-1]:
        data_y = np.array([])

        # filter only for points in the data, that are in the x borders of the aggretaed point
        if x == stepsX[1]:
            subdata = np.take(data, np.where(data[0] > x-stepsize/2)[0], axis=1)
        else:
            subdata = np.take(subdata, np.where(subdata[0] > x-stepsize/2)[0], axis=1)
        
        subdata_x = np.take(subdata, np.where(subdata[0] < x+stepsize/2)[0], axis=1)
        
        for y in stepsY[1:-1]:
                
            # filter only for points in the data, that are in the y borders of the aggretaed point
            if y == stepsY[1]:
                subdata_y = np.take(subdata_x, np.where(subdata_x[1] > y-stepsize/2)[0], axis=1)
            else:
                subdata_y = np.take(subdata_y, np.where(subdata_y[1] > y-stepsize/2)[0], axis=1)
                
            subdata_yy = np.take(subdata_y, np.where(subdata_y[1] < y+stepsize/2)[0], axis=1)

            if returnType == "array":
                if len(subdata_yy[0]) > 0:
                    data_agg = np.vstack((data_agg, np.array([x,y,np.mean(subdata_yy[2])]))) # append average z-value over aggregation point 
            
            elif returnType == "matrix":
                if len(subdata_yy[0]) > 0: 
                    data_y = np.append(data_y, np.mean(subdata_yy[2]))   
                elif len(data_y) > 0:
                    data_y = np.append(data_y, data_y[-1])
                elif len(data_matrix) > 0:
                    data_y = np.append(data_y, data_matrix[-1][len(data_y)])
                else:
                    data_y = np.append(data_y, 0)
        
        if returnType == "matrix":
            if len(data_matrix) == 0:
                data_matrix = data_y
            else:
                data_matrix = np.vstack((data_matrix, data_y))
        
    data_agg = data_agg[1:] # delete initialization point
    
    return data_agg if returnType == "array" else data_matrix.T

def applyAggFilter(data: np.array, filter_orientation: str="undefined"):
    '''
    applyFilter applies a filter to the point cloud to highlight the edges.
                           
    ______________________Implemented filters______________________________ 

    Laplace operator:       Scharr operator:        Scharr operator:
    [[-1, -1, -1],          [[ 47,  0,  -47],       [[ 47,  162,  47],
     [-1,  8, -1],           [162,  0, -162],        [  0,    0,   0]
     [-1, -1, -1]]           [ 47,  0,  -47]]        [-47, -162, -47]]
    (X/Y direction)           (X direction)            (Y direction)

    _________________________params/returns________________________________

                                                              [[(x1,y1,z), (x1,y2,z), ..., (x1,yn,z)], 
    :param data: aggregated scan as (n_x, n_y) shaped matrix:  [(x2,y1,z), ...                 :    ],
                                                               [    :                               ] 
                                                               [(xn,y1,z), (xn,y2,z), ..., (xn,yn,z)]]
    :param filter_orientation: edges orientation (x, y or undefined)

    :return: a aggregated (n_x, n_y) shaped np.array with values calculated 
             accordingly to the applied filter
    '''

    data_filter = np.array([]) # initialization of the return array
    
    # iteration over each aggregation point
    for x in range(1,len(data)-1):
        data_filter_y = np.array([])
        for y in range(1,len(data[0])-1):
            val_filter = 0 # initialization of the filter value for point (x,y)
            
            # iteration over (x,y) and all 8 neigbour points
            for i in range(-1, 2):
                for j in range(-1, 2):
                    # Laplace filter
                    # appply filter value based on the position to x,y (see method description)
                    if filter_orientation == "undefined":
                        if i == 0 and j == 0:
                            val_filter += 8 * data[y][x]
                        else:
                            val_filter += -1 * data[y+j][x+i]
                    
                    # Sobel filter in x direction (using the Scharr operator)
                    # appply filter value based on the position to x,y (see method description)
                    elif filter_orientation == "y":
                        if i != 0:
                            if i == -1:
                                if j == 0:
                                    val_filter += 162 * data[y+j][x+i]
                                else:
                                    val_filter += 47 * data[y+j][x+i]
                            else:
                                if j == 0:
                                    val_filter += -162 * data[y+j][x+i]
                                else:
                                    val_filter += -47 * data[y+j][x+i]
                    
                    # Sobel filter in y direction (using the Scharr operator)
                    # appply filter value based on the position to x,y (see method description)
                    elif filter_orientation == "x":
                        if j != 0:
                            if j == -1:
                                if i == 0:
                                    val_filter += 162 * data[y+j][x+i]
                                else:
                                    val_filter += 47 * data[y+j][x+i]
                            else:
                                if i == 0:
                                    val_filter += -162 * data[y+j][x+i]
                                else:
                                    val_filter += -47 * data[y+j][x+i]
            
            data_filter_y = np.append(data_filter_y, val_filter) # append the calculated filter value to the return array
        
        if x == 1:
            data_filter = data_filter_y
        else:
            data_filter = np.vstack((data_filter, data_filter_y))

    return data_filter.T

def maxPool(data: np.array, orientation: str="x"):
    '''
    maxPool applies max-pooling in a plus/minus 2 points area in each row or 
    colmun for each point to filter the important filter values in the area
                           
    ______________________________Max-Pooling_______________________________

                Input area:                  After Max-Pooling
                [[2.3, 10.4, -0.1],   ->     [[0, 10.4, 0],
    Example:    [ 4.5, 13.9,  3.4],   ->     [ 0, 13.9, 0]
                [ 1.2, 9.8,   5.6]]   ->     [ 0, 9.8,  0]]
                              (only take maximum 
                               value in row/col)
                
    ____________________________params/returns______________________________

                                                              [[(x1,y1,z), (x1,y2,z), ..., (x1,yn,z)], 
    :param data: filtered scan as (n_x, n_y) shaped matrix:    [(x2,y1,z), ...                 :    ],
                                                               [    :                               ] 
                                                               [(xn,y1,z), (xn,y2,z), ..., (xn,yn,z)]]
    :param orientation: edges orientation (x, y or undefined)

    :return: a aggregated (n_x, n_y) shaped np.array with values calculated 
             accordingly to the max pooling
    '''

    mean = np.mean(data)
    # iteration over each point
    for x in range(len(data)):
        for y in range(len(data[0])):
            if orientation == "y":
                try:
                    if np.abs(data[x][y]-mean) == np.max(np.abs(data[x, y-2:y+3]-mean)):
                        data[x][y] = np.abs(data[x][y]-mean)
                    else:
                        data[x][y] = 0
                except:
                    data[x][y] = 0
            else:
                try:
                    if np.abs(data[x][y]-mean) == np.max(np.abs(data[x-2:x+3, y]-mean)):
                        data[x][y] = np.abs(data[x][y]-mean)
                    else:
                        data[x][y] = 0
                except:
                    data[x][y] = 0
                               
    return data

def getBorders(data: np.array, stepsize: float=0.02, orientation: str="x"):
    '''
    getBodrders chooses the two most propable edges in the given orientation
    by calculaten the sum over each row/column and taking the two highest 
    values.
                
    ____________________________params/returns______________________________

                                                              [[(x1,y1,z), (x1,y2,z), ..., (x1,yn,z)], 
    :param data: pooled scan as (n_x, n_y) shaped matrix:      [(x2,y1,z), ...                 :    ],
                                                               [    :                               ] 
                                                               [(xn,y1,z), (xn,y2,z), ..., (xn,yn,z)]]
    :param stepsize: aggregation size
    :param orientation: edges orientation (x, y or undefined)

    :return: min and max border value plus the magnitude of the line sum
    '''

    if orientation == "x":
        lineSums = np.array([])
        for i in range(len(data[0])):
            lineSums = np.append(lineSums, np.sum(data[i]))

        border_1 = np.arange(0, 1.01, stepsize)[2:-2][np.where(lineSums == np.max(lineSums))[0]][0]
        border_2 = np.arange(0, 1.01, stepsize)[2:-2][np.where(lineSums == np.sort(lineSums)[-2])[0]][0]

        if border_1 > 0.9 and border_2 < 0.1: 
            border_1, border_2 = 1, 0
        elif border_1 < 0.1 and border_2 > 0.9:
            border_1, border_2 = 0, 1

        return min((border_1, border_2)), max((border_1, border_2)), (np.max(lineSums) + np.sort(lineSums)[-2])/2
    
    elif orientation == "y":
        lineSums = np.array([])
        for i in range(len(data)):
            lineSums = np.append(lineSums, np.sum(data[:,i]))
            
        border_1 = np.arange(0, 1.01, stepsize)[2:-2][np.where(lineSums == np.max(lineSums))[0]][0]
        border_2 = np.arange(0, 1.01, stepsize)[2:-2][np.where(lineSums == np.sort(lineSums)[-2])[0]][0]

        if border_1 > 0.9 and border_2 < 0.1: 
            border_1, border_2 = 1, 0
        elif border_1 < 0.1 and border_2 > 0.9:
            border_1, border_2 = 0, 1

        return min((border_1, border_2)), max((border_1, border_2)), (np.max(lineSums) + np.sort(lineSums)[-2])/2

def findAnomalie(data: np.array, stepsize: float=0.02):
    '''
    findAnomalie calculates the anomalie area by applying a filter + maxPool
    in x and y direction

    _________________________params/returns__________________________________

                                                              [[(x1,y1,z), (x1,y2,z), ..., (x1,yn,z)], 
    :param data: aggregated scan as (n_x, n_y) shaped matrix:  [(x2,y1,z), ...                 :    ],
                                                               [    :                               ] 
                                                               [(xn,y1,z), (xn,y2,z), ..., (xn,yn,z)]]
    :param stepsize: stepsize for aggregation

    :return: list with min/max x and y values of the anomalie: [xMin, xMax, yMin, yMax]
    '''

    data_agg = downScaling(data, stepsize=stepsize, returnType="matrix")

    data_filter = applyAggFilter(data_agg, filter_orientation="y")
    data_pool = maxPool(copy.deepcopy(data_filter), orientation="y")
    xMin, xMax, xMagnitude = getBorders(data_pool, stepsize=stepsize, orientation="y")

    data_filter = applyAggFilter(data_agg, filter_orientation="x")
    data_pool = maxPool(copy.deepcopy(data_filter), orientation="x")
    yMin, yMax, yMagnitude = getBorders(data_pool, stepsize=stepsize, orientation="x")

    if xMagnitude < yMagnitude: #and xMax - xMin < 0.2:
        xMin, xMax = 0, 1
    elif xMagnitude > yMagnitude: #and yMax - yMin < 0.2:
        yMin, yMax = 0, 1

    return [xMin, xMax, yMin, yMax]

def processFlow(ff: pd.DataFrame):
    # to array with single cycles
    data = ff.to_numpy().T
    cycleData = []
    for cycle in np.unique(data[2]):
        cycleData.append(np.take(data, np.where(data[2] == cycle)[0], axis=1))

    # calculate sums per time interval
    sumData = []
    fixedTimeStep = 500000000
    for i in range(len(cycleData)):
        print(i)
        sums = np.array([])
        aktIndex = 0
        aktTime = cycleData[i][3][aktIndex]
        while aktIndex < len(cycleData[i][0])-1:
            try:
                nextIndex = np.where(cycleData[i][3] >= aktTime + fixedTimeStep)[0][0]
            except:
                nextIndex = len(cycleData[i][0])-1
        
            sums = np.append(sums,np.sum(cycleData[i][1][aktIndex:nextIndex]))
        
            aktIndex = nextIndex
            aktTime = cycleData[i][3][aktIndex]

        #lowpass = scipy.signal.butter(10, 400, "lp", fs=1000, output='sos')
        #sums_filter = scipy.signal.sosfilt(lowpass, sums)
        sums_filter = sums

        m = np.mean(sums_filter)
        n_deletes = 0 
        while np.max(sums_filter[:8]) > 1.5 * m or np.min(sums_filter[:8]) < m / 1.5: #or sums_filter[1] > 1.5 * m or sums_filter[1] < m / 1.5 or sums_filter[2] > 1.5 * m or sums_filter[2] < m / 1.5:
            sums_filter = sums_filter[1:]
            n_deletes += 1
            if n_deletes > 40:
                break
        n_deletes = 0
        while np.max(sums_filter[-8:]) > 1.5 * m or np.min(sums_filter[-8:]) < m / 1.5: #or sums_filter[-2] > 1.5 * m or sums_filter[-2] < m / 1.5 or sums_filter[-3] > 1.5 * m or sums_filter[-3] < m / 1.5:
            sums_filter = sums_filter[:-1]
            n_deletes += 1
            if n_deletes > 40:
                break
    
        # standardize
        sums_filter_standard = (sums_filter - np.mean(sums_filter)) / np.std(sums_filter)

        n_points = 100
        sums_mean_n = np.array([])
        for i in range(n_points):
            sums_mean_n = np.append(sums_mean_n, np.mean(sums_filter_standard[int(i*(len(sums_filter_standard)/n_points)):int((i+1)*(len(sums_filter_standard)/n_points))]))    
        sums_mean_n_smooth = scipy.signal.savgol_filter(sums_mean_n, 10, 3)

        sums_std_n = np.array([])
        for i in range(n_points):
            sums_std_n = np.append(sums_std_n, np.std(sums_filter_standard[int(i*(len(sums_filter_standard)/n_points)):int((i+1)*(len(sums_filter_standard)/n_points))]))    
        sums_std_n_smooth = scipy.signal.savgol_filter(sums_std_n, 10, 3)

        sums_final = sums_mean_n_smooth * sums_std_n_smooth

        sumData.append([sums_filter_standard, sums_mean_n_smooth, sums_std_n_smooth, sums_final])
    return sumData

def getFlowBorders(X: np.array):
    borders = np.array([0,0])
    for cycle in X:
        sums, indexes = np.array([]), np.array([])
        plus, currentSum = True, 0
        for i in range(len(X[0])):
            if cycle[i] > 0.05 and plus == True or cycle[i] < -0.05 and plus == False:
                currentSum += cycle[i]
            else:
                sums = np.append(sums, currentSum)
                indexes = np.append(indexes, i-1)
                currentSum = cycle[i]
                plus = not plus

        sums = np.abs(sums)
        border = [indexes[np.where(sums == np.max(sums))[0]][0], indexes[np.where(sums == np.max(sums))[0]-1][0]]
        borders = np.vstack((borders, border))
    borders = borders[1:]
    return borders   

def getRMSE(x: np.array, y: np.array, index_split: int, modeltype: str="svr", hyperparam: list=[0.5, 0.005], getPreds: bool=False):
    if modeltype == "lr":
        
        RMSE_lr = []
        
        # X_min
        model = LinearRegression().fit(x[:index_split], y[:index_split,0])
        preds = {"X_min": model.predict(x[index_split:])}
        MSE = np.square(np.subtract(y[index_split:,0],model.predict(x[index_split:]))).mean() 
        RMSE = math.sqrt(MSE)
        RMSE_lr = [np.round(RMSE, 2)]
        # X_max
        model = LinearRegression().fit(x[:index_split], y[:index_split,1])
        preds["X_max"] = model.predict(x[index_split:])
        MSE = np.square(np.subtract(y[index_split:,1],model.predict(x[index_split:]))).mean() 
        RMSE = math.sqrt(MSE)
        RMSE_lr.append(np.round(RMSE, 2))

        if getPreds == True:
            return (RMSE_lr, preds)
        return RMSE_lr

    elif modeltype == "svr":
        C, gamma = hyperparam[0], hyperparam[1]

        RMSE_svr = []

        # X_min
        svr = svm.SVR(kernel="linear", C=C, gamma=gamma)
        svr.fit(x[:index_split], y[:index_split,0])
        preds = {"X_min": svr.predict(x[index_split:])}
        MSE = np.square(np.subtract(y[index_split:,0],svr.predict(x[index_split:]))).mean() 
        RMSE = math.sqrt(MSE)
        RMSE_svr = [np.round(RMSE, 2)]
        # X_max
        svr = svm.SVR(kernel="linear", C=C, gamma=gamma)
        svr.fit(x[:index_split], y[:index_split,1])
        preds["X_max"] = svr.predict(x[index_split:])
        MSE = np.square(np.subtract(y[index_split:,1],svr.predict(x[index_split:]))).mean() 
        RMSE = math.sqrt(MSE)
        RMSE_svr.append(np.round(RMSE, 2))
        # Y_min
        #svr = svm.SVR(kernel="linear", C=C, gamma=gamma)
        #svr.fit(x[:index_split], y[:index_split,2])
        #preds["Y_min"] = svr.predict(x[index_split:])
        #MSE = np.square(np.subtract(y[index_split:,2],svr.predict(x[index_split:]))).mean() 
        #RMSE = math.sqrt(MSE)
        #RMSE_svr.append(np.round(RMSE, 2))
        # Y_max
        #svr = svm.SVR(kernel="linear", C=C, gamma=gamma)
        #svr.fit(x[:index_split], y[:index_split,3])
        #preds["Y_max"] = svr.predict(x[index_split:])
        #MSE = np.square(np.subtract(y[index_split:,3],svr.predict(x[index_split:]))).mean() 
        #RMSE = math.sqrt(MSE)
        #RMSE_svr.append(np.round(RMSE, 2))
        # Extrusion
        #svr = svm.SVR(kernel="linear", C=0.5, gamma=0.005)
        #svr.fit(x[:index_split], E[:index_split])
        #MSE = np.square(np.subtract(E[index_split:],svr.predict(x[index_split:]))).mean() 
        #RMSE = math.sqrt(MSE)
        #RMSE_svr.append(RMSE)
        if getPreds == True:
            return (RMSE_svr, preds)
        return RMSE_svr

    elif modeltype == "lstm":
        X_train, Y_train = x[:index_split], y[:index_split] # np.hstack((Y,np.array([E]).T))
        X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
        X_train = np.asarray(X_train).astype(np.float32)
        Y_train = np.asarray(Y_train).astype(np.float32)
        X_test, Y_test = x[index_split:], y[index_split:]
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
        X_test = np.asarray(X_test).astype(np.float32)
        Y_test = np.asarray(Y_test).astype(np.float32)

        model = keras.Sequential()
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(64,return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[-1]))))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(20,return_sequences=False)))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(2))
        model.compile(loss='mse', optimizer='rmsprop', metrics=['mean_squared_error'])

        history = model.fit(x=X_train, y=Y_train, batch_size=20, epochs=60)

        RMSE_lstm, preds, labels = [], {}, ["X_min", "X_max"]
        for i in range(2):
            preds[labels[i]] = model.predict(X_test)[:,i]     
            MSE = np.square(np.subtract(Y_test[:,i],model.predict(X_test)[:,i])).mean() 
            RMSE = math.sqrt(MSE)
            RMSE_lstm.append(np.round(RMSE, 2))
        if getPreds == True:
            return (RMSE_lstm, preds)
        return RMSE_lstm

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
            if len(df.columns) == 2:
                i = 0
                for c in df.columns:
                    if i == 0:
                        df.rename(columns={c: "timestamp"}, inplace=True)
                    else:
                        df.rename(columns={c: "flow"}, inplace=True)
                    i += 1    
            else:
                i = 0
                for c in df.columns:
                    if i == 0:
                        df.rename(columns={c: "timestamp"}, inplace=True)
                    else:
                        df.rename(columns={c: f"x{i}"}, inplace=True)
                    i += 1

                df = df.drop(['x1'], axis=1)
                df = df[ df["x2"] != 0]
                df.rename(columns={"x2": "flow"}, inplace=True)
            
            df = df[df["flow"] != 0]
            #df = df[df["flow"] <= 2]
            df = df[df["flow"] >= -1]
    
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["timestamp"] = df["timestamp"].dt.strftime('%Y%m%d%H%M%S%f')
            df["timestamp"] = pd.to_datetime(df["timestamp"], format='%Y%m%d%H%M%S%f')
       
            df.reset_index(drop=True, inplace=True)

            data = np.array([df["timestamp"].astype(np.int64), np.array(df["flow"])]) # [Zeit in current ns, Filament Flow]
            data = np.vstack((data[:,1:], data[0,1:]-data[0,:-1])) # [Zeit in current ns, Filament Flow, Zeitdfferenz]
            
            # detect cycles
            cycle_index = np.array([])
            gaps = np.where(data[1] > 10)[0]

            for i in range(len(gaps)):
                #print(i)
                prev_index = 0 if i == 0 else gaps[i-1]
                cycle_index = np.append(cycle_index, np.ones(gaps[i]-prev_index)*i)
            cycle_index = np.append(cycle_index, np.ones(len(data[0])-gaps[-1])*len(gaps))
            data = np.vstack((data, cycle_index))

            # filter cycles
            data = data[:,np.in1d(data[3], np.unique(data[3])[np.where(np.unique(data[3], return_counts=True)[1] > 1000)[0]])]

            # reset cycle indexing
            cycle_index, index = np.array([]), 1
            for cycle in np.unique(data[3]):
                print(index)
                cycle_index = np.append(cycle_index, np.ones(len(np.where(data[3] == cycle)[0])) * index)
                index += 1
           
            # calculate datetime
            data = np.vstack((data, pd.to_datetime(data[0].astype(dtype=np.int64), utc=True, unit="ns"))) 
       
            df = pd.DataFrame({"timestamp": data[4], "flow": data[1], "cycle": cycle_index, "currrent-ms": data[0]})
            df = df[df["flow"] <= 2]
            df.reset_index(drop=True, inplace=True)

        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif "txt" in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=" ", names=["X","Y","Z"])
        
    except Exception as e:
        print(e)
        return None
    
    return df

def getDatabase():
    try:
        database = pd.read_csv("./" + "anomalies" + ".csv")
        database = database[["name", "xMin", "xMax", "yMin", "yMax", "cycle"]]
        database.rename(columns={"name": "Dateiname", "cycle": "Bauteil-Nr."}, inplace=True)
    except:
        database = pd.DataFrame({"Dateiname": [],"xMin": [],"xMax": [],"yMin": [],"yMax": [],"Bauteil-Nr.": []})
    return database

'''
_________________________________ APP _________________________________

made with Plotly Dash

Hints:
 - only use dash_bootstrap_components for uniform style

'''

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MORPH], suppress_callback_exceptions=True)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div([
    html.H2("FPQ", className="display-4"),
    html.Hr(),
    html.P(
        "TOOL zur ", className="lead"
    ),
    dbc.Nav(
        [
            dbc.NavLink("Handbuch", href="/H", active="exact"),
            dbc.NavLink("Kausales Modell", href="/CI", active="exact"),
        ],
        vertical=True,
        pills=True,
    ),
    html.Hr(),
    html.P(
        "DCPS Paket ", className="lead"
    ),
    dbc.Nav(
        [
            dbc.NavLink("Upload", href="/", active="exact"),
            dbc.NavLink("Scans verarbeiten", href="/ScanAnomalie", active="exact"),
            dbc.NavLink("Visualisierung", href="/visu", active="exact"),
            dbc.NavLink("Modell", href="/model", active="exact"),
            dbc.NavLink("G-Code-Modifier", href="/gcode", active="exact"),
        ],
        vertical=True,
        pills=True,
    ),
], style=SIDEBAR_STYLE)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content,
    dcc.Store(id="scan-storage", storage_type='local'),
    dcc.Store(id="scan-storage_temp1", storage_type='memory'),
    dcc.Store(id="scan-storage_temp2-0", storage_type='memory'),
    dcc.Store(id="scan-storage_temp2-1", storage_type='memory'),
    dcc.Store(id="scan-storage_temp2-2", storage_type='memory'),
    dcc.Store(id="scan-storage_temp2-3", storage_type='memory'),
    dcc.Store(id="scan-storage_temp2-4", storage_type='memory'),
    dcc.Store(id="scan-storage_temp3-0", storage_type='memory'),
    dcc.Store(id="scan-storage_temp3-1", storage_type='memory'),
    dcc.Store(id="scan-storage_temp3-2", storage_type='memory'),
    dcc.Store(id="scan-storage_temp3-3", storage_type='memory'),
    dcc.Store(id="scan-storage_temp3-4", storage_type='memory'),

    dcc.Store(id="raw-storage", storage_type='local'),
    dcc.Store(id="raw-storage_temp1", storage_type='memory'),
    dcc.Store(id="raw-storage_temp2", storage_type='memory'),

    # store part indexes
    dcc.Store(id="input-storage_1", storage_type='memory'),
    dcc.Store(id="input-storage_2", storage_type='memory'),
    dcc.Store(id="input-storage_3", storage_type='memory'),
    dcc.Store(id="input-storage_4", storage_type='memory'),
    dcc.Store(id="input-storage_5", storage_type='memory'),
    dcc.Store(id="input-storage_6", storage_type='memory'),
    dcc.Store(id="input-storage_7", storage_type='memory'),
    dcc.Store(id="input-storage_8", storage_type='memory'),
    dcc.Store(id="input-storage_9", storage_type='memory'),
    dcc.Store(id="input-storage_10", storage_type='memory'),
    dcc.Store(id="input-storage_11", storage_type='memory'),
    dcc.Store(id="input-storage_12", storage_type='memory'),
    dcc.Store(id="input-storage_13", storage_type='memory'),
    dcc.Store(id="input-storage_14", storage_type='memory'),
    dcc.Store(id="input-storage_15", storage_type='memory'),
    dcc.Store(id="input-storage_16", storage_type='memory'),

    dcc.Store(id="model-storage", storage_type='local'),
    dcc.Store(id="model-storage_temp1", storage_type='memory'),
    dcc.Store(id="model-storage_temp2", storage_type='memory'),

    dcc.Store(id="pred-storage", storage_type='local'),

    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
])


@app.callback(Output('scan-storage', 'data'),
              Input("scan-storage_temp1", 'data'),
              Input("scan-storage_temp2-0", 'data'),
              Input("scan-storage_temp2-1", 'data'),
              Input("scan-storage_temp2-2", 'data'),
              Input("scan-storage_temp2-3", 'data'),
              Input("scan-storage_temp2-4", 'data'),
              Input("scan-storage_temp3-0", 'data'),
              Input("scan-storage_temp3-1", 'data'),
              Input("scan-storage_temp3-2", 'data'),
              Input("scan-storage_temp3-3", 'data'),
              Input("scan-storage_temp3-4", 'data'))
def update_scanStorage(tmp1, tmp21, tmp22, tmp23, tmp24, tmp25, tmp31, tmp32, tmp33, tmp34, tmp35):
    print([tmp1, tmp21, tmp22, tmp23, tmp24, tmp25, tmp31, tmp32, tmp33, tmp34, tmp35])

    if tmp1 is not None:
        return tmp1
    elif tmp21 is not None:
        return tmp21
    elif tmp22 is not None:
        return tmp22
    elif tmp23 is not None:
        return tmp23
    elif tmp24 is not None:
        return tmp24
    elif tmp25 is not None: 
        return tmp25
    elif tmp31 is not None:
        print("1")
        return tmp31
    elif tmp32 is not None:
        print("2")
        return tmp32
    elif tmp33 is not None:
        print("3")
        return tmp33
    elif tmp34 is not None:
        print("4")
        return tmp34
    elif tmp35 is not None: 
        return tmp35
    else:
        raise PreventUpdate
    
@app.callback(Output('raw-storage', 'data'),
              Input("raw-storage_temp1", 'data'),
              Input("raw-storage_temp2", 'data'),
              State('raw-storage', 'data'))
def update_rawStorage(tmp1, tmp2, data):
    if tmp1 is None and tmp2 is None:
        raise PreventUpdate
    if tmp1 is not None:
        return tmp1
    else:
        return tmp2    

@app.callback(Output('model-storage', 'data'),
              Input("model-storage_temp1", 'data'),
              Input("model-storage_temp2", 'data'),
              State('model-storage', 'data'))
def update_modelStorage(tmp1, tmp2, data):
    if tmp1 is None and tmp2 is None:
        raise PreventUpdate
    if tmp1 is not None:
        return tmp1
    else:
        return tmp2  



# This is some **bold** text and this is some *italicized* text.

markdown_text_z0 = '''
#### Kausalen Inferenz Diagramm:
&nbsp
- Ein Kausales Inferenz Diagramm stellt eine kausale Beziehung zwischen Variablen. Wie die rechte Diagramm zeigt, es besteht aus Knoten, die Variablen darstellen, und gerichteten Pfeilen, die Pfeile zeigen die Richtung der Kausalität an.
- Die Zahlen auf den Pfeilen zeigen den Effekt der Veränderung der Quellvariable auf die Zielvariable an. Wenn die Zahl positiv ist, bedeutet dies, dass eine Erhöhung der Quellvariable auch eine Erhöhung der Zielvariable bewirkt. Wenn die Zahl negativ ist, bedeutet dies, dass eine Erhöhung der Quellvariable eine Verringerung der Zielvariable bewirkt. Die Stärke des Effekts wird normalerweise durch die Größe der Zahl angezeigt.
- Um ein Kausales Inferenz Diagramm zu verstehen, ist es wichtig zu beachten, dass es sich um eine Darstellung von Annahmen handelt, die auf Daten gestützt sind. Es ist nicht notwendigerweise eine Darstellung der wahren Kausalbeziehung zwischen den Variablen. Es ist wichtig, die Annahmen zu verstehen, die in das Diagramm einbezogen sind, und die Gültigkeit dieser Annahmen zu prüfen, bevor Schlussfolgerungen gezogen werden.
- Wenn Sie noch mehr Wissen möchten, können Sie die folgende Linke lesen: 
###### [1. Tutorial of LiNGAM algorithms](https://lingam.readthedocs.io/en/latest/tutorial/index.html)
###### [2. Causal Data Science](https://medium.com/causal-data-science/causal-data-science-721ed63a4027)
###### [3. Use causal graphs!](https://towardsdatascience.com/use-causal-graphs-4e3af630cf64)
'''

markdown_text_z1 = '''
# Fingerprints of Quality - Qualität 4.0 bei KMU
## Forschungsthema
### Ziel:
- Das Ziel des Vorhabens ist die Entwicklung und Nutzbarmachung von selbstlernenden 
Datenanalysealgorithmen auf Basis kausaler Inferenz zur Qualitätsabsicherung am 
Beispiel von Kleinserien und Unikaten bei KMU. Mit FPQ werden bestehende Methoden 
aufgegriffen, überarbeitet und neue Ansätze unter Qualitätsaspekten entwickelt und validiert. 
Das zu konzipierende Modell stützt sich auf Algorithmen, die anhand der Daten aus der laufenden Produktion Ursache-Wirkprinzipien mitlernen. Über ein Qualitäts-Assistenzsystem 
(AS) wird der Bediener durch die methodischen Schritte geführt, die es ihm ermöglichen, 
Versuchs- und Prozessdaten auszuwerten und in Beziehung zu setzen, Simulationen mit 
gerichteten Zusammenhängen durchzuführen, sein (a-priori) Prozesswissen algorithmisch in 
die Modellbildung einfließen zu lassen und die erhaltenen Analyseresultate zu kommunizieren. 
Das gesamtheitliche Ergebnis bildet die Grundlage zur iterativen Maßnahmenableitung, sodass 
die Produktqualität als Ergebnis der Prozessqualität bewahrt oder verbessert wird.
- Item 2...
- Item 3...

'''

markdown_text_z2 = '''
# Gesprächsnotizen
##   

### Mi. 05. 04:
- Modell zur kausalen Inferenz hinterlegen --> gerne dort wo man die anderen Modelle auswählen kann
- Visualisierung anhand eines Kausalgraphen im Zusammenhang mit dem Modell
##  

### Mi. 17. 04:
- Upload der csv-Datei 
- Als weitere Option einen Radnom Data Generator
- Auswahl der Variablen (eine y Variable und mehrere x Variablen)
- Modellerstellung mit den ausgewählten Variablen (Modell aus LinGAM Library)
- Visualisierung der entsprechenden Zusammenhängen anhand kausalem Graph
##  

### Mi. 19. 04:
- Arbeitszeit aufschreiben
- Die App sollte 3 Tabs beinhalten: 1. Tab = Handbuch, 2. Tab = Kausales Modell, 3. Tab = DCPS App (Anomalieerkennung)
'''


'''
_____________________________ home layout _________________________________

'''
home_layout = html.Div([
    dbc.Row([
        dbc.Col(
            html.Div(children=[dcc.Markdown(children=markdown_text_z1)],
                     className="p-3 bg-light rounded-3"),
            width=8),
        dbc.Col(
            html.Div(children=[dcc.Markdown(children=markdown_text_z2)],
                     className="p-3 bg-light rounded-3"),
            width=4),
    ])
])



'''
_____________________________ causal layout _______________________________

'''
options = [
    {'label': 'DirectLiNGAM', 'value': 'DL'},
    {'label': 'VARLiNGAM', 'value': 'VL'},
    {'label': 'RCD', 'value': 'RCD'}]

optionsplot = [    
    {'label': 'Bar', 'value': 'bar'},
    {'label': 'Line', 'value': 'line'},
    {'label': 'Scatter', 'value': 'Sc'}]

causal_layout = html.Div([
    # 1. Row
    dbc.Row([
        dbc.Col(
            html.Div(children=[html.H4("Daten hochladen", className="display-7"),
                               html.Hr(className="my-2"),
                               html.P("Lade Datei als .csv hoch:"),
                               dcc.Upload(
                                   id='upload-data',                                         
                                   children=html.Div(['Datei ablegen oder ', html.A('auswählen')]),                                          
                                   style={'width': '100%',
                                          'height': '60px',
                                          'lineHeight': '60px',
                                          'borderWidth': '1px',
                                          'borderStyle': 'dashed',
                                          'borderRadius': '5px',
                                          'textAlign': 'center'},                                          
                                   multiple=True)],
                     style={'width': '100%',
                            "position": "relative",
                            'display': 'inline-block',
                            'vertical-align': 'top',
                            "margin-right": "25px"},
                     className="h-100 p-3 bg-light rounded-3"),
            width=9),
        dbc.Col(
            html.Div(children=[html.H4('Random Nummers', className="display-7"),
                               html.Hr(className="my-2"),   
                               dbc.FormFloating([
                                   dbc.Input(id="input-row", type="number", min=1, max=500000, step=1), # id_2
                                   dbc.Label("Anzahl der Row:")],
                                   style={'margin-top': '10px', "margin-left": "0px", "fontSize": 18}),
                               dbc.FormFloating([
                                   dbc.Input(id="input-col", type="number", min=1, max=10, step=1), # id_3
                                   dbc.Label("Anzahl der Col:")],
                                   style={'margin-top': '10px', "margin-left": "0px", "fontSize": 18}),
                               dbc.Button(
                                   "Generieren",
                                   id="generate-button",
                                   size='sm',
                                   color="secondary",
                                   n_clicks=0,
                                   style={'margin-top': '20px', "margin-left": "0px", "fontSize": 18})],
                     style={'width': '100%',
                            "position": "relative",
                            'display': 'inline-block',
                            'vertical-align': 'top',
                            "margin-right": "25px"},
                     className="p-3 bg-light rounded-3"),
            width=3)
    ]),
    # 2. Row    
    dbc.Row([
        dbc.Col(
            html.Div(children=[dcc.Markdown(children=markdown_text_z0),
                               html.Hr(className="my-2"),
                               html.Br()], 
                     style={'width': '100%',
                            'position': 'relative',
                            'display': 'inline-block',
                            'vertical-align': 'top',
                            "margin-top": "25px",
                            "margin-right": "25px"},
                     className="p-3 bg-light rounded-3"),                     
            width=3),
        dbc.Col(
            html.Div(children=[html.H4('Kausale Inferenz ausgewählter Variablen / Daten visualisieren:'),
                               html.Hr(className="my-2"),
                               html.Br(),
                               dcc.Loading(
                                   children=[html.Div(id='output-div')], type="circle"),
                                html.Hr(),
                               html.Div(id='output-dia'),
                               html.Br()], 
                     style={'width': '100%',
                            'position': 'relative',
                            'display': 'inline-block',
                            'vertical-align': 'top',
                            "margin-top": "25px",
                            "margin-right": "25px"},
                     className="p-3 bg-light rounded-3"),                     
            width=6),
        dbc.Col(
            html.Div(children=[html.H4('Tabelle aus csv oder Randomsnummer:', className="display-7"),
                               html.Hr(className="my-2"),
                               html.Br(),
                               dcc.Loading(
                                   children=[html.Div(id='output-table')], type="circle"),
                               html.Hr(),
                               html.Br()],
                     style={'width': '100%',
                            "position": "relative",
                            'display': 'inline-block',
                            'vertical-align': 'top',
                            "margin-top": "25px",
                            "margin-right": "25px"},
                     className="p-3 bg-light rounded-3"),
            width=3)
    ])
])

@app.callback(
    Output('output-table', 'children'),
    Input('upload-data', 'contents'),
    Input('generate-button', 'n_clicks'),
    State('upload-data', 'filename'),
    State('input-row', 'value'),
    State('input-col', 'value') 
)

def data_source(x, y, a, b, c):
    if x is not None:
        output = update_output(x, a)
        x = None
        return output
    elif y is not None:
        output = update_table(y, b, c)
        y = None
        return output

# Table_Data
def update_output(list_of_contents, list_of_names):

    if list_of_contents is not None:
        children = [
            parse_content(c, n) for c, n in
            zip(list_of_contents, list_of_names)]
        return children

def parse_content(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        return html.Div(['There was an error processing this file.'])

    return html.Div([
        html.H5(filename),
        html.Hr(),
        dcc.Dropdown(
            id='xaxis-data',
            options=[{'label':x, 'value':x} for x in df.columns],
            placeholder='Input:',
            style={                       
                'borderRadius': '10px'},
            multi=True),
        dcc.Dropdown(
            id='yaxis-data',
            options=[{'label':x, 'value':x} for x in df.columns],
            placeholder='Output:',
            style={"margin-top": "25px", 'borderRadius': '10px'}),
        html.Br(),
        dbc.RadioItems(id='radiobutton',  options=options, inline=True),
        html.Br(),
        dbc.Button(id="submit-button",
                   size='sm',
                   color='secondary',
                   children="Erstellen",
                   style={'margin-top': '20px', "margin-left": "0px", "fontSize": 18}),
        html.Hr(),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            page_size=10,
            style_table={'overflowX': 'scroll'}),
        html.H4('Daten visualisieren:'),
        html.Hr(),
        dcc.Dropdown(
            id='xaxis-data-dia',
            options=[{'label':x, 'value':x} for x in df.columns],
            placeholder='Input:',
            style={                       
                'borderRadius': '10px'},
            multi=False),
        dcc.Dropdown(
            id='yaxis-data-dia',
            options=[{'label':x, 'value':x} for x in df.columns],
            placeholder='Output:',
            style={"margin-top": "25px", 'borderRadius': '10px'},
            multi=True),
        html.Br(),
        dbc.RadioItems(id='radiobutton2',  options=optionsplot, inline=True),        
        dbc.Button(id="submit-button2",
                   size='sm',
                   color='secondary',
                   children="Visualisieren",
                   style={'margin-top': '20px', "margin-left": "0px", "fontSize": 18}),        
        dcc.Store(id='stored-data', data=df.to_dict('records'))])

# Random_Data
def update_table(n_clicks, row, col):
    if n_clicks is not None:
        r = 1
        random_dict = {}
        
        for i in range(r):
            key = random.randint(0, 2)
            value_r = np.random.uniform(size=row)
            random_dict["x_" + str(key)] = value_r
        
        for i in range(col):
            if i == key:
                continue
            value = (random.randint(1, 10))*(random.choice(list(random_dict.values()))) + np.random.uniform(size=row)
            random_dict["x_" + str(i)] = value
        sorted_dict = {k: random_dict[k] for k in sorted(random_dict)}    
        df = pd.DataFrame.from_dict(sorted_dict)      
        df = df.round(5)
    
    return html.Div([
        dcc.Dropdown(
            id='xaxis-data',
            options=[{'label':x, 'value':x} for x in df.columns],
            placeholder='Input:',
            style={                       
                'borderRadius': '10px'},
            multi=True),
        dcc.Dropdown(
            id='yaxis-data',
            options=[{'label':x, 'value':x} for x in df.columns],
            placeholder='Output:',
            style={"margin-top": "25px", 'borderRadius': '10px'}),
        html.Br(),
        dbc.RadioItems(id='radiobutton',  options=options, inline=True),
        html.Br(),
        dbc.Button(id="submit-button",
                   size='sm',
                   color='secondary',
                   children="Erstellen",
                   style={'margin-top': '20px', "margin-left": "0px", "fontSize": 18}),
        html.Hr(),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            page_size=10,
            style_table={'overflowX': 'scroll'}),
        dcc.Store(id='stored-data', data=df.to_dict('records'))])      
        
@app.callback(
    Output('output-div', 'children'),
    Input('submit-button','n_clicks'),
    State('stored-data','data'),
    State('xaxis-data','value'),
    State('yaxis-data', 'value'),
    State('radiobutton', 'value')
)
def make_graphs(n, data, x_data, y_data, value):
    X = pd.DataFrame({x: [row[x] for row in data] for x in x_data})
    Y = [row[y_data] for row in data]
    XY_df = pd.concat([X, pd.DataFrame({y_data: Y})], axis=1)
    print(XY_df.head())

    if n is None:
        return dash.no_update
    elif value == 'DL':
        # DirectLiNGAM:
        model = lingam.DirectLiNGAM()
        model.fit(XY_df)
        labels = list(XY_df.columns)
        dot = make_dot(model.adjacency_matrix_, labels=labels)
    elif value == 'VL':
        # VARLiNGAM
        labels = list(XY_df.columns)  
        merged_labels = []
        labels_1 = [f'{labels[i]}(t)' for i in range(XY_df.shape[1])] 
        labels_2 = [f'{labels[i]}(t-1)' for i in range(XY_df.shape[1])]
        merged_labels.extend(labels_1)
        merged_labels.extend(labels_2)
        model = lingam.VARLiNGAM()
        model.fit(XY_df)
        dot = make_dot(
            np.hstack(model.adjacency_matrices_),
            ignore_shape=True,
            lower_limit=0.05,
            labels=merged_labels)    
    elif value == 'RCD':
        # RCD
        model = lingam.RCD()
        model.fit(XY_df)
        labels = list(XY_df.columns)  
        dot = make_dot(model.adjacency_matrix_, labels=labels)

    dot.format = 'png'
    dot.render('Causal_Inference')
    ci_png = 'Causal_Inference.png'
    ci_base64 = base64.b64encode(open(ci_png, 'rb').read()).decode('ascii')
    
    return html.Div([
        html.Img(
            src='data:image/png;base64,{}'.format(ci_base64))],
            style={'textAlign':'center', 'overflow': 'scroll'})

@app.callback(Output('output-dia', 'children'),
              Input('submit-button2','n_clicks'),
              State('stored-data','data'),
              State('xaxis-data-dia','value'),
              State('yaxis-data-dia', 'value'),
              State('radiobutton2', 'value')
)
def make_dia(n, data, x_data, y_data, value):
    if n is None:
        return dash.no_update
    elif value == 'bar':
        fig_dx = px.bar(data, x=x_data, y=y_data)
    elif value == 'line':
        fig_dx = px.line(data, x=x_data, y=y_data)
    elif value == 'Sc':
        fig_dx = px.scatter(data, x=x_data, y=y_data)
    return dcc.Graph(figure=fig_dx)

'''
_____________________________ upload page _______________________________

'''

up_layout = html.Div(id="upload-page", 
                     children=[html.H3("Daten hochladen", className="display-7"),
                               html.Hr(className="my-2"),
                               html.P("Lade rohe Scans als .txt oder rohe Filament-Flow Dateien als .csv hoch."),
                               dcc.Upload(id='upload-data', 
                                          children=html.Div(['Datei ablegen oder ', html.A('auswählen')]),
                                          style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '0px'},
                                          multiple=False)],
                     style={'width': '600px', 'display': 'inline-block', 'vertical-align': 'top', "border-radius": "25px", "margin-right": "35px"}, className="h-100 p-3 bg-light rounded-3")


@app.callback(
        output=dict(scan = Output('raw-storage_temp1', 'data'), 
                    c = Output('upload-page', 'children')),
        inputs=dict(content = Input('upload-data', 'contents')),
        state=dict(name = State('upload-data', 'filename'), 
                   l_m = State('upload-data', 'last_modified'),
                   data = State("raw-storage", "data"),
                   children = State('upload-page', 'children')))
def update_storage(content, name, l_m, data, children):
    if content is None:
        raise PreventUpdate
    print("start uploading process ...")
    if content is not None:
        # import file
        df = parse_contents(content, name, l_m)

        if df is None:
            raise PreventUpdate
        else:
            if ".csv" in name:
                newpath = "./dataStorage/filament-flow"
                if not os.path.exists(newpath):
                    os.makedirs(newpath)

                if not os.path.exists(newpath + "/" + "filament-flow.csv"):
                    df.to_csv(newpath + "/" + "filament-flow.csv", index=False)
                else:
                    ff = pd.read_csv(newpath + "/" + "filament-flow.csv", parse_dates=["timestamp"])
                    ff = ff[ff["timestamp"] < df["timestamp"].min()]
                    ff = ff[ff["timestamp"] > df["timestamp"].max()]
                    if len(ff) > 0:
                        df["cycle"] = np.array(df["cycle"]) + int(ff["cycle"].max())
                    ff = pd.concat((ff, df))
                    ff.sort_values(by=['timestamp'], ignore_index=True, inplace=True)
                    ff.to_csv(newpath + "/" + "filament-flow.csv", index=False)

            elif ".txt" in name:
                data = data or {"keine Daten vorhanden": []}
                data.pop('keine Daten vorhanden', None)

                i = 0
                while i < len(data.keys()):
                    k = list(data)[i]
                    try:
                        if not os.path.exists("./dataStorage/raw/scans/" + k.split("_")[-1] + "/" + "_".join(k.split("_")[:-1]) + ".csv"):
                            data.pop(k)
                        else:
                            i += 1
                    except:
                        data.pop(k)

                today = date.today()
                d = today.strftime("%d-%m-%Y")

                # create new folder(s) for saving the result
                newpath = "./dataStorage/raw/scans/" + d
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
               
                # filter the non usable points
                df = df[df["Z"] < 5]
                df = df[df["Z"] > -3]

                # find parts in scan
                data_border = delAllBorders(df)
                partsDims = findAllParts(data_border) # funktioniert nicht zuverlässig
                df = pd.DataFrame({"X": data_border[0], "Y": data_border[1], "Z": data_border[2]})
                df.to_csv("./dataStorage/raw/scans/" + d + "/" + name.split(".")[0] + ".csv", index=False)
                data.update({name.split(".")[0] + "_" + d: partsDims})

            children.append(dbc.Alert(f"Datei hochgeladen: {name}", color="success", style={"margin-top": "10px"}))
            print("done")
            return dict(scan = data, c = children)

'''
_____________________________ scan anomalie _______________________________

'''

progress = 0

scanAnomalie_layout = html.Div(id="scan-anomalie", children=[
    html.Div(children=[dcc.Dropdown(id='sa-dropdown', 
                                    options=[{'label': name, 'value': name} for name in ["keine Daten vorhanden"]],
                                    value=["keine Daten vorhanden"][0],
                                    style={'margin-bottom': '20px'}), 
                       html.Hr(className="my-2", style={'height': '1px', "margin-top": "0px"}),
                       dcc.Graph(id='sa-plot', style={'margin-top': '20px'}),
                       html.Div([dbc.Button("Scan entfernen?", id="open-deleteWindow", outline=True, color="secondary", n_clicks=0, style={"position": "relative", "float": "right",'margin-top': '10px', "fontSize": 18})]),
                       dbc.Modal(children=[dbc.ModalHeader(dbc.ModalTitle("Scan entfernen?"), close_button=True), 
                                           dbc.ModalBody("Falls der Scan bereits vorhanden ist oder kein Bauteil erkannt wurde, kannst du den Scan aus Datensatz löschen."),
                                           dbc.ModalFooter([html.Div([dbc.Button("Abbrechen", id="close", className="ms-auto", n_clicks=0, color="secondary", style={"margin-right": "10px"}),
                                                                      dbc.Button("Löschen", id="loesche", className="ms-auto", n_clicks=0, color="danger", style={"margin-right": "0px"})])])],
                                 id="modal-delete", centered=True, is_open=False)],
             style={'width': '600px', 'display': 'inline-block', 'vertical-align': 'top', "border-radius": "25px", "margin-right": "35px", "margin-bottom": "35px"}, className="h-100 p-3 bg-light rounded-3"),
    html.Div(id="inputBox", children=[], style={'width': '600px', 'display': 'inline-block', 'vertical-align': 'top', "border-radius": "25px"}, className="h-100 p-5")
])


@app.callback(
    dash.dependencies.Output('sa-dropdown', 'options'),
    [dash.dependencies.Input("raw-storage", "data")])
def update_dt_dropdown(data):
    if data is not None:
        data_dict = data
        return [{'label': i, 'value': i} for i in list(data_dict.keys())]
    
@app.callback(Output('sa-plot', 'figure'),
              Input('sa-dropdown', 'value'),
              State("raw-storage", "data"))
def update_graph(name, data):
    if not name == "keine Daten vorhanden":
        # load and aggregate raw scan
        raw_df = pd.read_csv("dataStorage/raw/scans/" + name.split("_")[-1] + "/" + "_".join(name.split("_")[:-1]) + ".csv")
        raw_data = raw_df.to_numpy().T
        agg_data = downScaling(raw_data, stepsize=1).T
        
        # display aggregated raw scan
        fig = px.scatter(x=agg_data[0], y=agg_data[1], color=agg_data[2], height=500, width=500, template="plotly_white")
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(xaxis_range=[-60,60], yaxis_range=[-60,60])

        # display current part index
        dims = data[name]
        for i in range(len(dims)):
            fig.add_annotation(x=dims[i][0]+(dims[i][1]-dims[i][0])/2, y=dims[i][2]+(dims[i][3]-dims[i][2])/2, 
                               text=str(i+1), font=dict(size=40, color="white"), 
                               showarrow=False)
            
        return fig
    fig = px.scatter(template="plotly_white", height=500, width=500)
    return fig
                                       
@app.callback(
    output=dict(c = Output('inputBox', 'children'), cn = Output('inputBox', 'className')),
    inputs=dict(name = Input('sa-dropdown', 'value')),
    state=dict(data = State("raw-storage", "data")))
def add_indexInputs(name, data):
    if not name == "keine Daten vorhanden":
        children = []
        dims = data[name]
        for i in range(len(dims)):
            el = html.Div([
                dbc.InputGroup([
                    dbc.InputGroupText(f"Scan {i+1}:"),
                    dbc.Input(id=f'input_index_{i+1}', placeholder="Index", type="number")], 
                    style={"margin-top": "10px"})
            ])
            children.append(el)
        children.append(html.Div([dbc.Button("Korrekt", color="success", id='assign-index', n_clicks=0, style={'padding': '5px 5px 5px 5px', "margin-top": "20px"})]))
        return dict(c=children, cn="h-100 p-5 bg-light border")
    else:
        raise PreventUpdate

for i in range(1,17):
    try:
        @app.callback(
            [Output(f'input-storage_{i}', 'data')],
            [Input(f'input_index_{i}', "value")],
            [State(f'input-storage_{i}', 'data')])
        def saveInputs(value, data):
            return [value]
    except:
        break      

@app.callback(
    output=dict(c = Output('scan-anomalie', 'children')),
    inputs=dict(n = Input('assign-index', 'n_clicks')),
    state=dict(children = State('scan-anomalie', 'children')))
def deleteInputs(n, children):
    if n is not None and not n == 0: 
        children.pop()
        children.append(html.Div(id="progressBox", children=[dbc.Progress(id="anomalie-progress")], style={'width': '600px', 'display': 'inline-block', 'vertical-align': 'top', "border-radius": "25px"}, className="h-100 p-3 bg-light rounded-3"))
        return dict(c=children)
    else:
        raise PreventUpdate

@app.callback(
    output=dict(storage = Output("scan-storage_temp1", "data")),
    inputs=dict(n = Input('assign-index', 'n_clicks')),
    state=dict(raw = State("raw-storage", "data"), 
               name = State('sa-dropdown', 'value'),
               data = State("scan-storage", "data"),
               indexes = [State('input-storage_1', 'data'), 
                          State('input-storage_2', 'data'),
                          State('input-storage_3', 'data'),
                          State('input-storage_4', 'data'), 
                          State('input-storage_5', 'data'),
                          State('input-storage_6', 'data'),
                          State('input-storage_7', 'data'),
                          State('input-storage_8', 'data'),
                          State('input-storage_9', 'data'),
                          State('input-storage_10', 'data'),
                          State('input-storage_11', 'data'),
                          State('input-storage_12', 'data'),
                          State('input-storage_13', 'data'),
                          State('input-storage_14', 'data'),
                          State('input-storage_15', 'data'), 
                          State('input-storage_16', 'data')]))
def processScan(n, raw, name, data, indexes):
    if n is not None and not n == 0:
        global progress
        print(indexes)

        today = date.today()
        d = today.strftime("%d-%m-%Y")

        newpath = "./dataStorage/scans/clean/" + d
        if not os.path.exists(newpath):
            os.makedirs(newpath)   

        data = data or {}
        anomalie_df = pd.DataFrame({"xMin": [], "xMax": [], "yMin": [], "yMax": [], "file": [], "name": [], "cycle": []})
        raw_df = pd.read_csv("dataStorage/raw/scans/" + name.split("_")[-1] + "/" + "_".join(name.split("_")[:-1]) + ".csv")
        dims = raw[name]

        progress += 5
        
        for i in range(1,int(np.nanmax(np.array(indexes, dtype=np.float64)))+1):
            df_part = copy.deepcopy(raw_df)
            df_part = df_part[df_part["X"] >= dims[indexes.index(i)][0]]
            df_part = df_part[df_part["X"] <= dims[indexes.index(i)][1]]
            df_part = df_part[df_part["Y"] >= dims[indexes.index(i)][2]]
            df_part = df_part[df_part["Y"] <= dims[indexes.index(i)][3]]
            progress += int(95/len(dims)/8)

            df_part = clearPart(df_part, newpath + "/", "_".join(name.split("_")[:-1]) + "_" + str(i) + ".csv")
            print("Part " + str(i) + " cleared.")
            progress += int((95/len(dims)/8)*6)

            df_part.to_csv(newpath + "/" + "_".join(name.split("_")[:-1]) + "_" + str(i) + ".csv", index=False)
            
            print("Part " + str(i) + " done. Saved the file as " + "_".join(name.split("_")[:-1]) + "_" + str(i) + ".csv")
            
            print("Searching anomalie: Part" + str(i) + " ...")
            part_data = df_part.to_numpy().T

            best_b = findAnomalie(part_data, stepsize=0.02) 
            
            if best_b[1] - best_b[0] < 0.8 or best_b[3] - best_b[2] < 0.8:
                data["_".join(name.split("_")[:-1]) + "_" + str(i) + "_" + d] = best_b
                best_b = pd.DataFrame({"xMin": [best_b[0]], "xMax": [best_b[1]], "yMin": [best_b[2]], "yMax": [best_b[3]]})
                best_b["file"] = [newpath + "/" + "_".join(name.split("_")[:-1]) + "_" + str(i) + ".csv"]
                best_b["name"] = ["_".join(name.split("_")[:-1]) + "_" + str(i)]
                best_b["cycle"] = [None]
                anomalie_df = pd.concat([anomalie_df, best_b], ignore_index=True)
                print("Anomalie found: Part", i)
            else:
                data["_".join(name.split("_")[:-1]) + "_" + str(i) + "_" + d] = None
                print("No Anomalie found: Part", i)
            
            progress += int(95/len(dims)/8)

        try:
            anomalie_df_loaded = pd.read_csv("./" + "anomalies" + ".csv")
    
            for index, row in anomalie_df.iterrows():
                if not row["name"] in list(anomalie_df_loaded["name"]):
                    anomalie_df_loaded = pd.concat([anomalie_df_loaded, pd.DataFrame({"xMin": [row["xMin"]], "xMax": [row["xMax"]], "yMin": [row["yMin"]], "yMax": [row["yMax"]], "file": [row["file"]], "name": [row["name"]], "cylce": [row["cycle"]]})], ignore_index=True)
                
            anomalie_df_loaded.to_csv("./" + "anomalies" + ".csv", index=False)
        except:
            anomalie_df.to_csv("./" + "anomalies" + ".csv", index=False)
        
        print(data)
        progress = 100
        return dict(storage=data)
    else:
        raise PreventUpdate

try:
    @app.callback(
        output=dict(value = Output('anomalie-progress', 'value'), label = Output('anomalie-progress', 'label')),
        inputs=dict(n = Input('interval-component', 'n_intervals')))
    def showProgress(n):
        return dict(value=progress, label=f"{progress} %" if progress >= 5 else "")
except:
    pass

@app.callback(
    output=dict(c = Output('progressBox', 'children')),
    inputs=dict(p = Input('anomalie-progress', 'value')),
    state=dict(children = State('progressBox', 'children')))
def showProgressDone(p, children):
    if p == 100:
        children.pop()
        children.append(dbc.Alert('Gesäuberte Scans + Fehler gespeichert! Gehe zur Seite "Visualisierung" um dir die Ergebnise anzeigen zu lassen.', color="success"))
        return dict(c=children)
    else:
        raise PreventUpdate

@app.callback(Output('modal-delete', 'is_open'),
              Input("open-deleteWindow", 'n_clicks'),
              Input("close", 'n_clicks'),
              Input("loesche", 'n_clicks'),
              State('modal-delete', 'is_open'))
def update_modal(n1, n2, n3, is_open):
    if n1 or n2 or n3:
        return not is_open
    return is_open

@app.callback(Output('raw-storage_temp2', 'data'),
              Input("loesche", 'n_clicks'),
              State('sa-dropdown', 'value'),
              State("raw-storage", "data"))
def update_scanTemp2(n, name, data):
    if n == 0:
        raise PreventUpdate
    if n:
        try:
            os.remove("dataStorage/raw/scans/" + name.split("_")[-1] + "/" + "_".join(name.split("_")[:-1]) + ".csv")
        except:
            pass

        del data[name]
        return data
    else:
        return None
        
'''
__________________________ visualization ____________________________

'''

visu_layout = html.Div([dbc.Row([dbc.Col(html.Div(id="flow-layout",
                                                  children=[dbc.Button('+ Flow', id='add-graphF', n_clicks=0, outline=True, color="secondary", style={"font-size":25, "padding-left":10, "padding-right":10, "padding-top":15, "padding-bottom":17, "line-height":10, "text-indent":0}),
                                                            html.Td(id='-data', style={'display': 'inline-block', 'vertical-align': 'top', 'padding': '0px 20px 20px 20px'})]), width=6),
                                 dbc.Col(html.Div(id="scanner-layout",
                                                  children=[dbc.Button('+ Scan', id='add-graphS', n_clicks=0, outline=True, color="secondary", style={"font-size":25, "padding-left":10, "padding-right":10, "padding-top":15, "padding-bottom":17, "line-height":10, "text-indent":0}),
                                                            html.Td(id='-dataS', style={'display': 'inline-block', 'vertical-align': 'top', 'padding': '0px 20px 20px 20px'})]), width=6)], justify="start"),
                       ])

# Filament-Flow:
@app.callback(
    Output('flow-layout', 'children'),
    Input('add-graphF', 'n_clicks'),
    State('flow-layout', 'children'))
def add_graph(valAdd, children):
    if valAdd:
        valAdd = valAdd-1
        if os.path.exists("./dataStorage/filament-flow" + "/" + "filament-flow.csv"):
            ff = pd.read_csv("./dataStorage/filament-flow" + "/" + "filament-flow.csv", parse_dates=["timestamp"])
            dateSelect = dcc.DatePickerRange(id=f'date-picker_{valAdd}', 
                                             min_date_allowed=ff["timestamp"].min() - datetime.timedelta(days=1), 
                                             max_date_allowed=ff["timestamp"].max(), 
                                             initial_visible_month=ff["timestamp"].min(), 
                                             style={'margin-bottom': '10px'},)
        else:
            dateSelect = None

        el = html.Div(children=[dateSelect,
                                html.Hr(className="my-2", style={'height': '1px', "margin-top": "0px"}),
                                dcc.Graph(id=f'time-series_{valAdd}', style={'margin-top': '20px'}),
                                dbc.Row([dbc.Col(html.Div([dbc.RadioItems(options=[{"label": "zeige rohe FF-Daten", "value": 1}, 
                                                                                   {"label": "zeige Summe über feste Zeit", "value": 2},
                                                                                   {"label": "zeige verarbeitetes Signal", "value": 3}],
                                                                          value=1, id=f"radio-input_{valAdd}", switch=False, style={'margin-top': '10px', "fontSize": 18})]), width=5),
                                         dbc.Col(html.Div([dbc.InputGroup([dbc.Button("Scan zuordnen", id=f"match_{valAdd}", n_clicks=0, outline=False, color="success"),
                                                                           dbc.Input(id=f"cycle_{valAdd}", placeholder="Druck-Nr.")], style={'margin-top': '70px', "fontSize": 18})]), width=4)], justify="between"),
                                dbc.Toast(children=[], id=f"match-info_{valAdd}", header="Info", duration=10000, is_open=False, icon="success", dismissable=True, style={"fontSize": 18, "position": "fixed", "top": 10, "right": 10, "width": 400})],
                      style={'width': '900px', 'display': 'inline-block', 'vertical-align': 'top', "border-radius": "25px", "margin-right": "35px", "margin-top": "35px"}, className="h-100 p-3 bg-light rounded-3")
        children.append(el)
        return children
    else:
        raise PreventUpdate

for i in range(5):
    try:
        @app.callback(Output(f'time-series_{i}', 'figure'),
                      Input(f'date-picker_{i}', 'start_date'),
                      Input(f'date-picker_{i}', 'end_date'),
                      Input(f"radio-input_{i}", 'value'))
        def update_graph(start, end, value):
            if start is not None and end is not None:
                ff = pd.read_csv("./dataStorage/filament-flow" + "/" + "filament-flow.csv", parse_dates=["timestamp"])
                
                ff = ff[ff["timestamp"] >= pd.to_datetime(start, utc=True, unit="ns")] 
                ff = ff[ff["timestamp"] <= pd.to_datetime(end, utc=True, unit="ns")]
                
                if value == 1:
                    fig = px.line(ff, x="timestamp", y="flow", color="cycle", template="plotly_white", height=600)
                    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"), yaxis=dict(fixedrange=False))
                else:
                    sumData = processFlow(ff)

                if value == 2:
                    fig = go.Figure()
                    for i in range(len(sumData)):
                        fig.add_trace(go.Scatter(x=np.arange(0, len(sumData[i][0]), 1), y=sumData[i][0], mode='lines', name=f'Plot_{i+ff["cycle"].min()}'))
                    fig.update_layout(template="plotly_white", height=600)
                
                elif value == 3:
                    fig = go.Figure()
                    for i in range(len(sumData)):
                        fig.add_trace(go.Scatter(x=np.arange(0, len(sumData[i][3]), 1), y=sumData[i][3], mode='lines', name=f'Plot_{i+ff["cycle"].min()}'))
                    fig.update_layout(template="plotly_white", height=600)

                return fig
            
            fig = px.line(template="plotly_white", height=600)
            return fig
    except:
        break

# Scans:
@app.callback(
    Output('scanner-layout', 'children'),
    [Input('add-graphS', 'n_clicks')],
    [State('scanner-layout', 'children')],)
def add_graph(valAdd, children):
    if valAdd:
        valAdd = valAdd-1
        el = html.Div(children=[dcc.Dropdown(id=f'scanner-dropdown_{valAdd}',
                                             options=[{'label': name, 'value': name} for name in ["keine Daten vorhanden"]],
                                             value=["keine Daten vorhanden"][0], 
                                             style={'margin-bottom': '20px'}), 
                                html.Hr(className="my-2", style={'height': '1px', "margin-top": "0px"}),
                                dcc.Graph(id=f'scanner-plot_{valAdd}', style={'margin-top': '20px'}),
                                dbc.Row([dbc.Col(html.Div([dbc.Switch(id=f"anomalie-switch_{valAdd}", label="Zeige erkannte Anomalie", value=False, style={'margin-top': '15px', "fontSize": 18})]), width=4),
                                         dbc.Col(html.Div([dbc.Button("Rotieren", id=f"rotate_{valAdd}", outline=True, color="secondary", n_clicks=0, style={'margin-top': '10px', "margin-left": "0px", "fontSize": 18}),
                                                           dbc.Button("Falsch erkannt?", id=f"open-deleteWindow_{valAdd}", outline=True, color="secondary", n_clicks=0, style={'margin-top': '10px', "margin-left": "10px", "fontSize": 18})]), width="auto")], justify="between"),
                                dbc.Modal(children=[dbc.ModalHeader(dbc.ModalTitle("Falsch erkannt?"), close_button=True), 
                                                    dbc.ModalBody("Falls die Anomalie falsch erkannt wurde, kannst du das Bauteil aus dem erstellten Datensatz löschen."),
                                                    dbc.ModalFooter([html.Div([dbc.Button("Abbrechen", id=f"close_{valAdd}", className="ms-auto", n_clicks=0, color="secondary", style={"margin-right": "10px"}),
                                                                               dbc.Button("Löschen", id=f"loesche_{valAdd}", className="ms-auto", n_clicks=0, color="danger", style={"margin-right": "0px"})])])],
                                          id=f"modal-delete_{valAdd}", centered=True, is_open=False),
                                dbc.Toast(children=[], id=f"anomalie-info_{valAdd}", header="Info", duration=10000, is_open=False, style={"margin-top": "10px", "fontSize": 18})], 
                      style={'width': '900px', 'display': 'inline-block', 'vertical-align': 'top', "border-radius": "25px", "margin-right": "35px", "margin-top": "35px"}, className="h-100 p-3 bg-light rounded-3")
        children.append(el)
        return children
    else:
        raise PreventUpdate

for i in range(5):
    try:
        @app.callback(
            dash.dependencies.Output(f'scanner-dropdown_{i}', 'options'),
            dash.dependencies.Input("scan-storage", "data")) #, prevent_initial_call=True
        def update_plot_dropdown(data):
            if data is not None:
                key_list = []
                for k in list(data.keys()):
                    if os.path.exists("./dataStorage/scans/clean/" + k.split("_")[-1] + "/" + "_".join(k.split("_")[:-1]) + ".csv"):
                        key_list.append(k)
                return [{'label': i, 'value': i} for i in key_list]
    except:
        break

for i in range(5):
    try:
        @app.callback(output=dict(graph = Output(f'scanner-plot_{i}', 'figure'),
                                  info = Output(f"anomalie-info_{i}", "children"),
                                  open = Output(f"anomalie-info_{i}", "is_open"),
                                  temp = Output(f'scan-storage_temp3-{i}', 'data')),
                      inputs=dict(name = Input(f'scanner-dropdown_{i}', 'value'),
                                  switch = Input(f'anomalie-switch_{i}', 'value'),
                                  button = Input(f"rotate_{i}", "n_clicks")),
                      state=dict(data = State("scan-storage", "data")))
        def update_graph(name, switch, button, data):
            if not name == "keine Daten vorhanden":
                plot_df = pd.read_csv("./dataStorage/scans/clean/" + name.split("_")[-1] + "/" + "_".join(name.split("_")[:-1]) + ".csv", parse_dates=True)
                
                if ("rotate_0" == ctx.triggered_id or "rotate_1" == ctx.triggered_id or "rotate_2" == ctx.triggered_id or "rotate_3" == ctx.triggered_id or "rotate_4" == ctx.triggered_id) and button != 0:
                    print("!!")
                    plot_df = rotatePart(plot_df, "./dataStorage/scans/clean/" + name.split("_")[-1] + "/", "_".join(name.split("_")[:-1]))
                
                    plot_data = downScaling(plot_df.to_numpy().T, stepsize=0.01).T

                    fig = px.scatter(x=plot_data[0], y=plot_data[1], color=plot_data[2], height=800, width=800, template="plotly_white")
                    fig.update_traces(marker=dict(size=9))
                    if switch == True:
                        if data[name] is None:
                            return dict(graph = fig, info = html.P("Keine Anomalie erkannt."), open = True, temp = None)
                        data[name] = [data[name][2], data[name][3], 1-data[name][1], 1-data[name][0]]
                        fig.add_shape(type="rect", x0=data[name][0], y0=data[name][2], x1=data[name][1], y1=data[name][3], line=dict(color="Red")) 
                        return dict(graph = fig, info = html.P("Der rote Kasten zeigt die Anomalie."), open = True, temp = data)
                    data[name] = [data[name][2], data[name][3], 1-data[name][1], 1-data[name][0]]
                    return dict(graph = fig, info = [], open = False, temp = data)
                else:
                    plot_data = downScaling(plot_df.to_numpy().T, stepsize=0.01).T

                    fig = px.scatter(x=plot_data[0], y=plot_data[1], color=plot_data[2], height=800, width=800, template="plotly_white")
                    fig.update_traces(marker=dict(size=9))
                    if switch == True:
                        if data[name] is None:
                            return dict(graph = fig, info = html.P("Keine Anomalie erkannt."), open = True, temp = None)
                        fig.add_shape(type="rect", x0=data[name][0], y0=data[name][2], x1=data[name][1], y1=data[name][3], line=dict(color="Red")) 
                        return dict(graph = fig, info = html.P("Der rote Kasten zeigt die Anomalie."), open = True, temp = None)
                    return dict(graph = fig, info = [], open = False, temp = None)
            
            fig = px.scatter(template="plotly_white", height=800, width=800)
            return dict(graph = fig, info = [], open = False, temp = None)
    except:
        break

for i in range(5):
    try:
        @app.callback(Output(f'modal-delete_{i}', 'is_open'),
                      Input(f"open-deleteWindow_{i}", 'n_clicks'),
                      Input(f"close_{i}", 'n_clicks'),
                      Input(f"loesche_{i}", 'n_clicks'),
                      State(f'modal-delete_{i}', 'is_open'))
        def update_modal(n1, n2, n3, is_open):
            if n1 or n2 or n3:
                return not is_open
            return is_open
    except:
        break

for i in range(5):
    try:
        @app.callback(Output(f'scan-storage_temp2-{i}', 'data'),
                      Input(f"loesche_{i}", 'n_clicks'),
                      State(f'scanner-dropdown_{i}', 'value'),
                      State("scan-storage", "data"))
        def update_scanTemp2(n, name, data):
            if n:
                os.remove("./dataStorage/scans/clean/" + name.split("_")[-1] + "/" + "_".join(name.split("_")[:-1]) + ".csv")

                try:
                    anomalie_df = pd.read_csv("./" + "anomalies" + ".csv")
                    anomalie_df.drop([anomalie_df[anomalie_df["name"] == "_".join(name.split("_")[:-1])].index], inplace=True)
                    anomalie_df.to_csv("./" + "anomalies" + ".csv", index=False)
                except:
                    pass

                del data[name]
                return data
            else:
                return None
    except:
        break

# match both
for i in range(5):
    try:
        @app.callback(output=dict(info = Output(f'match-info_{i}', 'children'),
                                  is_open = Output(f"match-info_{i}", "is_open")),
                      inputs=dict(n = Input(f'match_{i}', 'n_clicks')),
                      state=dict(cycle = State(f'cycle_{i}', 'value'),
                                 name = State(f'scanner-dropdown_{i}', 'value')))
        def show_Info(n, cycle, name):
            if n is not None and cycle is not None and name is not None:
                try:
                    anomalie_df = pd.read_csv("./" + "anomalies" + ".csv")
                    anomalie_df.at[anomalie_df[anomalie_df["name"] == "_".join(name.split("_")[:-1])].index[0], "cycle"] = cycle
                    anomalie_df.to_csv("./" + "anomalies" + ".csv", index=False)
                except:
                    return dict(info = html.P("Zuordnung nicht erfolgreich."), is_open = True)

                return dict(info = html.P(f"Scan erfolgreich zu Druck {cycle} zugeordnet."), is_open = True)
            raise PreventUpdate    
    except:
        break

'''
__________________________ model ____________________________

'''

model_layout = html.Div([html.Div(id="model-actions",
                                  children=[html.H3("Modell-Aktionen", className="display-7", style={"width": "auto", 'display': 'inline-block', "margin-right": "30px"}),
                                            html.Hr(className="my-2"),
                                            dbc.Button("trainieren", id="train", color="success", n_clicks=0, style={'margin-top': '10px', "margin-left": "10px", "fontSize": 18}),
                                            dbc.Button("testen", id="test", color="warning", n_clicks=0, style={'margin-top': '10px', "margin-left": "10px", "fontSize": 18}),
                                            dbc.Button("vorhersagen", id="predict", color="primary", n_clicks=0, style={'margin-top': '10px', "margin-left": "10px", "fontSize": 18}),
                                            dbc.Button("optimieren", id="optimize", color="info", n_clicks=0, style={'margin-top': '10px', "margin-left": "10px", "fontSize": 18})],
                                  style={'width': "230px", 'display': 'inline-block', 'vertical-align': 'stretch', "border-radius": "25px", "margin": "20px"}, className="h-100 p-3 bg-light rounded-3"),
                         html.Div(id="model-summary",
                                  children=[html.H3("Modell-Übersicht", className="display-7", style={"width": "auto", 'display': 'inline-block', "margin-right": "30px"}),
                                            dbc.Select(id="model-select", options=[{"label": "LR", "value": "1"},
                                                                                   {"label": "SVR", "value": "2"},
                                                                                   {"label": "LSTM", "value": "3"}], value="1",
                                                       style={"width": "300px", 'display': 'inline-block', "float": "right", "fontSize": 18}),
                                            html.Hr(className="my-2"),
                                            html.Div(id="modell-info")],
                                  style={'width': "40%", 'display': 'inline-block', 'vertical-align': 'top', "border-radius": "25px", "margin": "20px"}, className="h-100 p-3 bg-light rounded-3"),
                         html.Div(id="model-result", 
                                  children=[html.H3("Modell-Ergebnis", className="display-7"), 
                                            html.Hr(className="my-2")],
                                  style={'width': "900px", 'display': 'block', 'vertical-align': 'top', "border-radius": "25px", "margin": "20px"}, className="h-100 p-3 bg-light rounded-3"),
                         html.Div(id="database", 
                                  children=[html.H3("Datenspeicher", className="display-7", style={"width": "auto", 'display': 'inline-block', "margin-right": "30px"}), 
                                            dbc.Button("Update", id="update", outline=True, color="secondary", n_clicks=0, style={"width": "100px", 'display': 'inline-block', "float": "right", "fontSize": 18}),
                                            html.Hr(className="my-2"),
                                            dbc.Table.from_dataframe(getDatabase(), striped=True, bordered=True, hover=True)],
                                  style={'width': "900px", 'display': 'block', 'vertical-align': 'top', "border-radius": "25px", "margin": "20px"}, className="h-100 p-3 bg-light rounded-3")], 
                        style={"width": "100%"})

@app.callback(output=dict(c = Output('database', 'children')),
              inputs=dict(value = Input('update', 'n_clicks')),
              state=dict(children = State('database', 'children')))
def show_modellInfos(value, children): 
    if value is not None and value != 0:
        while len(children) > 3:
            children.pop()
        children.append(dbc.Table.from_dataframe(getDatabase(), striped=True, bordered=True, hover=True))
        return dict(c = children)
    raise PreventUpdate

@app.callback(output=dict(c = Output('model-summary', 'children'),
                          ms = Output("model-storage_temp1", "data")),
              inputs=dict(value = Input('model-select', 'value'),
                          result_c = Input('model-result', 'children')),
              state=dict(train = State('train', 'n_clicks'),
                         children = State('model-summary', 'children'),
                         storage = State("model-storage", "data"),
                         storage_temp = State("model-storage_temp2", "data")))
def show_modellInfos(value, result_c, train, children, storage, storage_temp):
    print("show") 
    
    if value is not None:
        while len(children) > 3:
            children.pop()

        try:
            if storage == None or ("Hinzugefügte" in str(result_c[-1]["props"]["children"]) and 'model-result' == ctx.triggered_id) or ("Gamma" in str(result_c[-1]["props"]["children"]) and 'model-result' == ctx.triggered_id):
                X = np.load('./dataStorage/model/X.npy', allow_pickle=True)
                Y = np.load('./dataStorage/model/Y.npy', allow_pickle=True)
                #E = np.load('./dataStorage/model/E.npy', allow_pickle=True)

                index_split = int(len(X)*(3/4))

                borders = getFlowBorders(X)

                # linear Regression
                RMSE_lr = getRMSE(borders, Y, index_split, "lr")

                # SVR
                if not storage_temp == None:
                    C = storage_temp["C"]
                    gamma = storage_temp["gamma"]
                elif not storage == None:
                    C = storage["C"]
                    gamma = storage["gamma"]
                else:
                    C = 0.5
                    gamma = 0.005

                RMSE_svr = getRMSE(borders, Y, index_split, "svr", [C, gamma])
            
                # LSTM
                RMSE_lstm = getRMSE(X, Y, index_split, "lstm")
            
                # save
                storage = {"n_points": len(X), "RMSE_lr": RMSE_lr, "RMSE_svr": RMSE_svr, "RMSE_lstm": RMSE_lstm, "C": C, "gamma": gamma}
        except:
            pass

        if value == "1":
            n = storage["n_points"]
            RMSE_x = np.round((storage["RMSE_lr"][0]+storage["RMSE_lr"][1])/2, 2)
            children.extend([html.P("Aufbau: Multiple lineare Regession", style={"fontSize": 18}), 
                             html.P(f"Anzahl der Datenpunkte im Modell: {n}", style={"fontSize": 18}),
                             html.P(f"RMSE für vorhersage der Grenzen: {RMSE_x}", style={"fontSize": 18}),
                             ])
        elif value == "2":
            n = storage["n_points"]
            RMSE_x = np.round((storage["RMSE_svr"][0]+storage["RMSE_svr"][1])/2, 2)
            #RMSE_y = np.round((storage["RMSE_svr"][2]+storage["RMSE_svr"][3])/2, 2)
            #r2_e = storage["RMSE_svr"][4]
            C = storage["C"]
            gamma = storage["gamma"]
            children.extend([html.P("Aufbau: Support Vector Regession mit linearer Kernel-Funktion", style={"fontSize": 18}),
                             html.P(f"Anzahl der Datenpunkte im Modell: {n}", style={"fontSize": 18}),
                             html.P(f"RMSE für vorhersage der Grenzen: {RMSE_x}", style={"fontSize": 18}),
                             #html.P(f"RMSE für vorhersage der Y-Grenzen: {RMSE_y}", style={"fontSize": 18}),
                             #html.P(f"R^2 für vorhersage der Extrusion: {r2_e}", style={"fontSize": 18}),
                             html.P(f"Hyperparameter: C={C}, gamma={gamma}", style={"fontSize": 18})
                             ])
        else:
            n = storage["n_points"]
            RMSE_x = np.round((storage["RMSE_lstm"][0]+storage["RMSE_lstm"][1])/2, 2)
            #RMSE_y = np.round((storage["RMSE_lstm"][2]+storage["RMSE_lstm"][3])/2, 2)
            #r2_e = storage["RMSE_lstm"][4]
            children.extend([html.P("Aufbau: Netzwerk mit zwei bidirectionalen LSTM-Layers mit Dropout-Layers getrennt.", style={"fontSize": 18}),
                             html.P(f"Anzahl der Datenpunkte im Modell: {n}", style={"fontSize": 18}),
                             html.P(f"RMSE für vorhersage der Grenzen: {RMSE_x}", style={"fontSize": 18}),
                             #html.P(f"RMSE für vorhersage der Y-Grenzen: {RMSE_y}", style={"fontSize": 18}),
                             #html.P(f"R^2 für vorhersage der Extrusion: {r2_e}", style={"fontSize": 18}),
                             ])
            
        return dict(c=children, ms=storage)
    raise PreventUpdate

@app.callback(output=dict(c = Output('model-result', 'children'),
                          ms = Output("model-storage_temp2", "data"),
                          ps = Output("pred-storage", "data")),
              inputs=dict(train = Input('train', 'n_clicks'),
                          test = Input('test', 'n_clicks'),
                          predict = Input('predict', 'n_clicks'),
                          optimize = Input('optimize', 'n_clicks')),
              state=dict(storage = State("model-storage", "data"),
                         value = State('model-select', 'value'),
                         children = State('model-result', 'children'),
                         pred_storage = State("pred-storage", "data")))
def modelActions(train, test, predict, optimize, storage, value, children, pred_storage):
    print("modelAction")
    
    while len(children) > 2:
        children.pop()
        print("pop")
    
    if "train" == ctx.triggered_id and train != 0:
        X = np.load('./dataStorage/model/X.npy', allow_pickle=True)
        Y = np.load('./dataStorage/model/Y.npy', allow_pickle=True)

        try:
            database = pd.read_csv("./" + "anomalies" + ".csv")
            database = database[["name", "xMin", "xMax", "yMin", "yMax", "cycle"]]
            database.rename(columns={"name": "Dateiname", "cycle": "Bauteil-Nr."}, inplace=True)
            database = database[database["Bauteil-Nr."].notnull()]
        except:
            database = pd.DataFrame({"Dateiname": [],"xMin": [],"xMax": [],"yMin": [],"yMax": [],"Bauteil-Nr.": []})
        
        ff = pd.read_csv("./dataStorage/filament-flow" + "/" + "filament-flow.csv", parse_dates=["timestamp"])    
        ff = ff[ff['cycle'].isin(list(database["Bauteil-Nr."]))]      
        sumData = processFlow(ff)

        addedCycles = []
        for i in range(len(sumData)):
            if not sumData[i][3] in X:
                X = np.vstack((X, sumData[i][3]))
                Y = np.vstack((Y, database.to_numpy().T[[1,2]].T[i]))
                addedCycles.append(list(ff["cycle"])[i])
                print("add Point")

        np.save('./dataStorage/model/X.npy', X)
        np.save('./dataStorage/model/Y.npy', Y)
        
        children.append(html.P("Hinzugefügte Bauteile: " + str(addedCycles), style={"fontSize": 18}))
        return dict(c=children, ms=None, ps=pred_storage)
    
    elif "test" == ctx.triggered_id and test != 0:
        X = np.load('./dataStorage/model/X.npy', allow_pickle=True)
        Y = np.load('./dataStorage/model/Y.npy', allow_pickle=True)

        try:
            database = pd.read_csv("./" + "anomalies" + ".csv")
            database = database[["name", "xMin", "xMax", "yMin", "yMax", "cycle"]]
            database.rename(columns={"name": "Dateiname", "cycle": "Bauteil-Nr."}, inplace=True)
            database = database[database["Bauteil-Nr."].notnull()]
        except:
            database = pd.DataFrame({"Dateiname": [],"xMin": [],"xMax": [],"yMin": [],"yMax": [],"Bauteil-Nr.": []})
        
        ff = pd.read_csv("./dataStorage/filament-flow" + "/" + "filament-flow.csv", parse_dates=["timestamp"])    
        ff = ff[ff['cycle'].isin(list(database["Bauteil-Nr."]))]      
        sumData = processFlow(ff)
            
        for i in range(len(sumData)):
            X = np.vstack((X, sumData[i][3]))
            Y = np.vstack((Y, database.to_numpy().T[[1,2]].T[i]))

        index_split = int(len(X)-len(sumData))

        # LR
        if value == "1":
            borders = getFlowBorders(X)
            RMSE = getRMSE(borders, Y, index_split, "lr", getPreds=False)

        # SVR
        if value == "2":
            borders = getFlowBorders(X)
            RMSE = getRMSE(borders, Y, index_split, "svr", [storage["C"], storage["gamma"]], getPreds=False)

        # LSTM
        elif value == "3":
            RMSE = getRMSE(X, Y, index_split, "lstm", getPreds=False)

        children.append(html.P("Im Durchschnitt hatte das Modell einen RMSE von: " + str(np.round((RMSE[0] + RMSE[1]) / 2, 2)) + " für die Daten im Speicher.", style={"fontSize": 18}))
        #children.append(html.P("Im Durchschnitt hatte das Modell einen RMSE in Y-Richtung von: " + str(np.round((RMSE[2] + RMSE[3]) / 2, 2)) + " für die Daten im Speicher.", style={"fontSize": 18}))
        return dict(c=children, ms=None, ps=pred_storage)

    elif "predict" == ctx.triggered_id and predict != 0:
        X = np.load('./dataStorage/model/X.npy', allow_pickle=True)
        Y = np.load('./dataStorage/model/Y.npy', allow_pickle=True)

        try:
            database = pd.read_csv("./" + "anomalies" + ".csv")
            database = database[["file", "name", "xMin", "xMax", "yMin", "yMax", "cycle"]]
            database.rename(columns={"name": "Dateiname", "cycle": "Bauteil-Nr."}, inplace=True)
            database = database[database["Bauteil-Nr."].notnull()]
        except:
            database = pd.DataFrame({"Dateiname": [],"xMin": [],"xMax": [],"yMin": [],"yMax": [],"Bauteil-Nr.": []})
        
        ff = pd.read_csv("./dataStorage/filament-flow" + "/" + "filament-flow.csv", parse_dates=["timestamp"])    
        ff = ff[ff['cycle'].isin(list(database["Bauteil-Nr."]))]      
        sumData = processFlow(ff)
            
        for i in range(len(sumData)):
            X = np.vstack((X, sumData[i][3]))
            Y = np.vstack((Y, database.to_numpy().T[[2,3]].T[i]))

        index_split = int(len(X)-len(sumData))

        # LR
        if value == "1":
            modeltype = "lr"
            borders = getFlowBorders(X)
            RMSE, preds = getRMSE(borders, Y, index_split, "lr", getPreds=True)

        # SVR
        if value == "2":
            modeltype = "svr"
            borders = getFlowBorders(X)
            RMSE, preds = getRMSE(borders, Y, index_split, "svr", [storage["C"], storage["gamma"]], getPreds=True)

        # LSTM
        elif value == "3":
            modeltype = "lstm"
            RMSE, preds = getRMSE(X, Y, index_split, "lstm", getPreds=True)

        if pred_storage == None:
            pred_storage = {}
        
        print(preds)
        for i in range(len(list(database["Dateiname"]))):
            pred_storage[list(database["Dateiname"])[i] + "_" + list(database["file"])[i].split("/")[-2] + "_" + modeltype] = [preds["X_min"][i], preds["X_max"][i], 0, 1]
        
        names = []
        for i in range(len(list(database["Dateiname"]))):
            names.append(list(database["Dateiname"])[i] + "_" + list(database["file"])[i].split("/")[-2])

        children.extend([dcc.Dropdown(id='pred-dropdown',
                                      options=[{'label': name, 'value': name} for name in names],
                                      value=names[0], 
                                      style={'margin-bottom': '20px'}), 
                         html.Hr(className="my-2", style={'height': '1px', "margin-top": "0px"}),
                         dcc.Graph(id='pred-plot', style={'margin-top': '20px'}),
                         dbc.Switch(id="anomalie-switch", label="Zeige erkannte Anomalie", value=False, style={'margin-top': '15px', "fontSize": 18}),
                         dbc.Switch(id="pred-switch", label="Zeige vorhergesagte Anomalie", value=False, style={'margin-top': '15px', "fontSize": 18})])
        #children.append(html.P("Vorhersagen: " + str(preds) + " für die Daten im Speicher.", style={"fontSize": 18}))
        return dict(c=children, ms=None, ps=pred_storage)

    elif "optimize" == ctx.triggered_id and optimize != 0:
        if value == "1":
            children.append(html.P("Hyperparameter-Optimierung nicht verfügbar für lineare Regression.", style={"fontSize": 18}))
            return dict(c=children, ms=None, ps=pred_storage)
        elif value == "2":
            X = np.load('./dataStorage/model/X.npy', allow_pickle=True)
            Y = np.load('./dataStorage/model/Y.npy', allow_pickle=True)
            borders = getFlowBorders(X)

            param = {'C' : [storage["C"]-storage["C"]/2, storage["C"]-storage["C"]/5, storage["C"], storage["C"]+storage["C"]/5, storage["C"]+storage["C"]/2],'gamma' : [storage["gamma"]-storage["gamma"]/2, storage["gamma"]-storage["gamma"]/5, storage["gamma"], storage["gamma"]+storage["gamma"]/5, storage["gamma"]+storage["gamma"]/2]},
            grid = GridSearchCV(svm.SVR(kernel="linear"), param_grid=param, cv=5, n_jobs = -1, verbose = 2)
            
            bestC, bestGamma = None, None
            for i in range(2):
                grid.fit(borders, Y[:,i])
                if bestC == None:
                    bestC = grid.best_params_["C"]  
                    bestGamma = grid.best_params_["gamma"]
                else:
                    bestC = (bestC + grid.best_params_["C"])/2
                    bestGamma = (bestGamma + grid.best_params_["gamma"])/2

            children.append(html.P("C wurde von " + str(storage["C"]) + " auf " + str(bestC) + " optimiert.", style={"fontSize": 18}))
            children.append(html.P("Gamma wurde von " + str(storage["gamma"]) + " auf " + str(bestGamma) + " optimiert.", style={"fontSize": 18}))
            storage["C"] = bestC
            storage["gamma"] = bestGamma
            return dict(c=children, ms=storage, ps=pred_storage)
        elif value == "3":
            children.append(html.P("Hyperparameter-Optimierung noch nicht verfügbar für das LSTM-Modell.", style={"fontSize": 18}))
            return dict(c=children, ms=None, ps=pred_storage)
    
    else:
        raise PreventUpdate

try:
    @app.callback(output=dict(graph = Output('pred-plot', 'figure')),
                  inputs=dict(name = Input('pred-dropdown', 'value'),
                              switch_a = Input('anomalie-switch', 'value'),
                              switch_p = Input('pred-switch', 'value')),
                  state=dict(data_p = State("pred-storage", "data"),
                             data_s = State("scan-storage", "data"),
                             value = State('model-select', 'value')))
    def update_graph(name, switch_a, switch_p, data_p, data_s, value):
        if not name == None:
            print(name)
            plot_df = pd.read_csv("./dataStorage/scans/clean/" + name.split("_")[-1] + "/" + "_".join(name.split("_")[:-1]) + ".csv", parse_dates=True)
            plot_data = downScaling(plot_df.to_numpy().T, stepsize=0.01).T

            fig = px.scatter(x=plot_data[0], y=plot_data[1], color=plot_data[2], height=800, width=800, template="plotly_white")
            fig.update_traces(marker=dict(size=9))
            if switch_a == True:
                fig.add_shape(type="rect", x0=data_s[name][0], y0=data_s[name][2], x1=data_s[name][1], y1=data_s[name][3], line=dict(color="Red")) 
            if switch_p == True:
                if value == "1":
                    modelType = "lr"
                elif value == "2":
                    modelType = "svr"
                elif value == "3":
                    modelType = "lstm"
                fig.add_shape(type="rect", x0=data_p[name + "_" + modelType][0], y0=data_p[name + "_" + modelType][2], x1=data_p[name + "_" + modelType][1], y1=data_p[name + "_" + modelType][3], line=dict(color="Green")) 

            return dict(graph = fig)
            
        fig = px.scatter(template="plotly_white", height=800, width=800)
        return dict(graph = fig)
except:
    pass


'''
__________________________ g-code ____________________________

'''

gcode_layout = html.Div([html.Div(id="inputs", 
                                  children=[html.H3("G-Code-Modifier", className="display-7"),
                                            html.Hr(className="my-2"),
                                            html.P("Der G-Code Modifier bietet die Möglichkeit fehlerhaftete Wände nach Benutzereingaben zu erzeugen."),
                                            html.Td(),
                                            dbc.FormFloating([dbc.Input(id="part_count", type="number", min=1, max=50, step=1), dbc.Label("Anzahl modifizierter Teile")], style={'margin-top': '10px', "margin-left": "0px", "fontSize": 18}),                                                      dbc.FormFloating([dbc.Input(id="layersize", type="number", min=0.1, max=0.3, step=0.1), dbc.Label("Layerstärke in mm")], style={'margin-top': '10px', "margin-left": "0px", "fontSize": 18}),
                                            dbc.FormFloating([dbc.Input(id="exStrength", type="number", min=0.5, max=3, step=0.1), dbc.Label("Extrusionsstärke")], style={'margin-top': '10px', "margin-left": "0px", "fontSize": 18}),
                                            dbc.Tooltip("genormt auf normalen Vorschub (= 1)", target="exStrength", placement="right"),
                                            dbc.RadioItems(id="exType", options=[{"label": "Überextrusion", "value": 1}, {"label": "Unterextrusion", "value": 2}], value=1, inline=True, style={'margin-top': '10px', "margin-left": "0px", "fontSize": 18}),
                                            dbc.Checklist(id="random", options=[{"label": "Zufällig anordnen", "value": 1}], value=[1], style={'margin-top': '10px', "margin-left": "0px", "fontSize": 18}),
                                            html.Div(id="boxInputs", children=[], style={'margin-top': '10px', "margin-left": "0px", "fontSize": 18}),
                                            html.Div(id="sizeInputs",
                                                     children=[dbc.Label("Min/Max Größe des Fehlers", id="size-label", html_for="range-slider", style={'margin-top': '10px', "margin-left": "20px", "fontSize": 18}),
                                                               dbc.Tooltip(r"in % der Fläche des Bauteils", target="size-label", placement="right"),
                                                               dcc.RangeSlider(id="size-slider", min=5, max=50, value=[10, 20], tooltip={"placement": "bottom", "always_visible": True})], 
                                                     style={"display": "block"}),
                                            html.Div(id="rangeInputs",
                                                     children=[dbc.Label("X-Begrenzung", id="X-label", html_for="range-slider", style={'margin-top': '10px', "margin-left": "20px", "fontSize": 18}),
                                                               dbc.Tooltip(r"in % der Seitenlänge des Bauteils", target="X-label", placement="right"),
                                                               dcc.RangeSlider(id="x-slider", min=0, max=100, value=[40, 50], tooltip={"placement": "bottom", "always_visible": True}),
                                                               dbc.Label("Y-Begrenzung", id="Y-label", html_for="range-slider", style={'margin-top': '10px', "margin-left": "20px", "fontSize": 18}),
                                                               dbc.Tooltip(r"in % der Seitenlänge des Bauteils", target="Y-label", placement="right"),
                                                               dcc.RangeSlider(id="y-slider", min=0, max=100, value=[50, 60], tooltip={"placement": "bottom", "always_visible": True})], 
                                                     style={"display": "none"}),
                                            dbc.Button("G-Code erzeugen", id="create", outline=True, color="primary", n_clicks=0, style={'margin-top': '20px', "margin-left": "0px", "fontSize": 18}),
                                            html.Div(id="output", children=[], style={'margin-top': '20px'})],
                                  style={'width': '900px', 'display': 'inline-block', 'vertical-align': 'top', "border-radius": "25px", "margin-right": "35px", "margin-top": "35px"}, className="h-100 p-3 bg-light rounded-3")])

@app.callback(output=dict(sI = Output('sizeInputs', 'style'),
                          rI = Output('rangeInputs', 'style')),
              inputs=dict(value = Input('random', 'value')))
def show_boxInputs(value):
    if value is not None:
        if len(value) == 1:
            return dict(sI={"display": "block"}, rI={"display": "none"})
        return dict(sI={"display": "none"}, rI={"display": "block"})
    raise PreventUpdate

@app.callback(output=dict(c = Output("output", "children")),
              inputs=dict(create = Input("create", "n_clicks")),
              state=dict(count = State('part_count', 'value'),
                         layersize = State('layersize', 'value'),
                         mode = State('exType', 'value'),
                         doRandom = State('random', 'value'),
                         size = State('size-slider', 'value'),
                         xBorders = State('x-slider', 'value'),
                         yBorders = State('y-slider', 'value'),
                         expression = State('exStrength', 'value')))
def gCodeMod(create, count, layersize, mode, doRandom, size, xBorders, yBorders, expression):
 
    #count     : Anzahl modifiziert Teile
    #fpath_in   : Pfad der originalen .gcode Datei
    #bbox_parameter: relative Y und Z Koordinaten der Fehlerstelle mit Komma getrennt (z.B: 0.5,0.6,0.1,0.2) ODER 'random_bbox'
    #mode       : Über/Unterextrusion (over/under)
    #expression : relative Grad der Fehler (z.B 0.4)
    #fpath_out  : Ausgangsordner zum Speichern der produzierten GCode Dateien, wenn leer dann neues Ordner mit heutigen Datum erstellen.
    #min_threshold : minimale relative Fläche des Bounding Boxed falls bbox_parameter = 'random_bbox' (default 0.05)
    #max_threshold : maximale relative Fläche des Bounding Boxed falls bbox_parameter = 'random_bbox' (default 0.2)

    if not create == None and not create == 0: 

        fpath_in = ""
        if layersize == 0.1:
            fpath_in += "./dataStorage/G-Code/UM3_01.gcode"
        elif layersize == 0.2:
            fpath_in += "./dataStorage/G-Code/UM3_02.gcode"
        elif layersize == 0.3:
            fpath_in += "./dataStorage/G-Code/UM3_03.gcode"
    
        if mode == 1:
            mode = "over"
        elif mode == 2:
            mode = "under"
    
        if len(doRandom) == 1:
            bbox_parameter = "random_bbox"
            min_threshold = size[0] / 100
            max_threshold = size[1] / 100
        else:
            bbox_parameter = str(xBorders[0]/100) + "," + str(xBorders[1]/100) + "," + str(yBorders[0]/100) + "," + str(yBorders[1]/100)

        fpath_out = "./G-Code/"
        if os.path.exists(fpath_out + str(date.today())):
            fpath_out = fpath_out + str(date.today()) + "/"
        elif not os.path.exists(fpath_out + str(date.today())):
            os.makedirs(fpath_out + str(date.today()))
            fpath_out = fpath_out + str(date.today()) + "/"

        orig_filename = fpath_in.split('/')[-1]
        orig_filename = orig_filename.split('.gcode')[0]
   
        if orig_filename[:3] == '04_':
            orig_filename = orig_filename.split('_',1)[-1]
   
        # Produce the wanted number of modified g-codes
        for i in range(int(count)):
            # initiate processor class
            proc = GCodeProcessor()
            with open(fpath_in, 'r') as f:
                proc.parse(f.read())
            min_x,max_x,min_y,max_y,min_z,max_z = proc.get_min_max_xyz()
            area_tot = round(abs((max_y-min_y)*(max_z-min_z)),2)
            min_threshold = 0.05 * area_tot 
            max_threshold = 0.25 * area_tot
       
            # Produce uniformally distributed bounding box parameters.
            # Area under bounding box should respect a minimal and maximal area threshold, as well as a tolerance value to avoid wall edges.
            if bbox_parameter == 'random_bbox':
                #print('Randomized Bounding Box:')
                tol = 1
                area_bbox = 0
           
                #print('begin randomization..')
                while area_bbox < min_threshold or area_bbox > max_threshold:
                    print(min_y)
                    rand_y = [round(random.uniform(min_y + tol,max_y - tol),3),round(random.uniform(min_y + tol,max_y - tol),3)]
                    rand_z = [round(random.uniform(min_z + tol,max_z - tol),3),round(random.uniform(min_z + tol,max_z - tol),3)]
                    area_bbox = round(abs((rand_y[0] - rand_y[1])*(rand_z[0]-rand_z[1])),2)
                  
                bbox = [
                    {'X': min_x, 'Y': min(rand_y), 'Z': min(rand_z)},
                    {'X': max_x, 'Y': max(rand_y), 'Z': max(rand_z)}
                    ]
                print(bbox) 
                print('Randomizierte bounding box: ',bbox)  
           
            else:
                bbox = bbox_parameter.split(',')
                bbox = [float(i) for i in bbox]
            
                range_y = max_y - min_y
                range_z = max_z - min_z
                bbox = [min_x, bbox[0]*range_y + min_y, bbox[2]*range_z + min_z, 
                        max_x,  bbox[1]*range_y + min_y, bbox[3]*range_z + min_z]
           
                area_bbox = round(abs((bbox[4] - bbox[1])*(bbox[5]-bbox[2])),2)
                print('die Fläche der Fehler beträgt ca. ' , round(area_bbox / area_tot *100, 1), '% der gesamt Fläches des Wandes')

                bbox = [
                    {'X': min_x, 'Y': bbox[1], 'Z': bbox[2]},
                    {'X': max_x, 'Y': bbox[4], 'Z': bbox[5]}
                    ] 
                print('Bounding box: ', bbox)  

            if mode == 'over':
                if expression == '':
                    fac = round(random.uniform(0.4,1.2),1)
                else: 
                    fac = float(expression)
                do_overextrusion(bbox, proc, factor= fac)
            elif mode == 'under':
                if expression == '':
                    fac = round(random.uniform(0.5,0.9),1)
                else: 
                    fac = float(expression)
                do_underextrusion(bbox, proc, factor = fac)
            else:
                print('please choose only underextrusion or overextrusion')
                exit()
       
            bbox_tag = ['',round(bbox[0]['Y'],1),round(bbox[1]['Y'],1),round(bbox[0]['Z'],1),round(bbox[1]['Z'],1)]
            bbox_tag = '_'.join([str(i).replace('.','^') for i in bbox_tag])
       
            # create and save file
            mod_filename = '_'.join([orig_filename, mode, str(int(fac*100))]) + bbox_tag + '.gcode'
            mod_filename = fpath_out + '/' + mode + '/' + mod_filename 
            if not os.path.exists(fpath_out + mode) and not os.path.exists(os.path.abspath(fpath_out + "/"+ mode)):
                os.makedirs(fpath_out + mode)
       
            log = 'modified GCode file: ' + mod_filename
            log += '\r\n' 
            log += 'original GCode file: ' + mod_filename
            with open(mod_filename, 'w') as f:
                print('creating new gcode file: ', mod_filename)
                f.write(proc.synthesize())
            f.close()
        return dict(c = dbc.Alert("Es wurden erfolgteich alle G-Code Dateien erstellt. Du findest sie unter " + fpath_out + ".", color="success", style={"margin-top": "20px"}))
    raise PreventUpdate

'''
__________________________ page content ____________________________

'''

@app.callback(
        Output("page-content", "children"), 
              [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/H":
        return home_layout
    elif pathname == "/CI":
        return causal_layout
    if pathname == "/":
        return up_layout
    elif pathname == "/ScanAnomalie":
        return scanAnomalie_layout
    elif pathname == "/visu":
        return visu_layout
    elif pathname == "/model":
        return model_layout
    elif pathname == "/gcode":
        return gcode_layout

    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )

'''
__________________________ main ____________________________

'''

if __name__ == '__main__':
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    if path != os.getcwd():
        os.chdir(path)
    
    app.run_server(debug=False, port=8077, threaded=True)
    #app.config["suppress_callback_exceptions"] = True