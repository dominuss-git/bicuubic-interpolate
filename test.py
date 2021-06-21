import cv2
import numpy as np
import math
import sys
import time
from tkinter.filedialog import askopenfilename
import os

# Interpolation kernel
def cubic_interpolate(s, a):
  if (abs(s) >= 0) & (abs(s) <= 1):
    return (a + 2) * (abs(s) ** 3) - (a + 3) * (abs(s) ** 2) + 1

  if (abs(s) > 1) & (abs(s) <= 2):
    return a * (abs(s) ** 3) - (5 * a) * (abs(s) ** 2) + (8 * a) * abs(s) - 4 * a
  return 0

#Paddnig
def padding(img, H, W, C):
  zimg = np.zeros((H + 4, W + 4, C))
  zimg[2 : H + 2, 2: W + 2, : C] = img
  #Pad the first/last two col and row
  zimg[2 : H + 2, 0 : 2, : C] = img[ :, 0 : 1, : C]
  zimg[H + 2 : H + 4, 2 : W + 2, : ] = img[H - 1 : H, :, :]
  zimg[2 : H + 2, W + 2 : W + 4, : ] = img[:, W - 1 : W, :]
  zimg[0 : 2, 2 : W + 2, : C] = img[0 : 1, :, : C]
  #Pad the missing eight points
  zimg[0 : 2, 0 : 2, : C] = img[0, 0, : C]
  zimg[H + 2 : H + 4, 0 : 2, : C] = img[H - 1, 0, : C]
  zimg[H + 2 : H + 4, W + 2 : W + 4, : C] = img[H - 1, W - 1, : C]
  zimg[0 : 2, W + 2 : W + 4, : C] = img[0, W - 1, : C]
  return zimg

def get_progressbar_str(progress, start, dist, base):
  MAX_LEN = 30
  BAR_LEN = int(MAX_LEN * progress)

  tim = time.time() - start
  around = (1 / progress * tim)

  a_sec = math.floor(around)
  around -= a_sec
  a_min = math.floor(a_sec / 60) 
  a_sec -= a_min * 60
  a_hour = math.floor(a_min / 60)
  a_min -= a_hour * 60

  sec = math.floor(tim)
  tim -= sec
  min = math.floor(sec / 60) 
  sec -= min * 60
  hour = math.floor(min / 60)
  min -= hour * 60

  if progress * 100 % 2 == 0:
    print(' ok!')
    cv2.imwrite(f'bicubic_{progress * 100}_{base}', dist)

  return (
    'Progress:[' + '=' * BAR_LEN +
    ('>' if BAR_LEN < MAX_LEN else '') +
    ' ' * (MAX_LEN - BAR_LEN) +
    '] %.1f%%' % (progress * 100.) + ' %02d' % hour + ':%02d' %
    min + ':%02d' % sec + ':' + 
    '%02d' % int(round(tim, 2) * 10 ** 2) + '  ( ~ %02d' % a_hour + ":%02d" %
    a_min + ':%02d' % a_sec + ')'
  )

# Bicubic operation
def bicubic(img, ratio, a, base):
  #Get image size
  H, W, C = img.shape

  img = padding(img, H, W, C)
  #Create new image
  dH = math.floor(H * ratio)
  dW = math.floor(W * ratio)
  dst = np.zeros((dH, dW, 3))

  h = 1 / ratio

  print('Start bicubic interpolation')
  print('It will take a little while...')
  print(H, W, C, dH, dW, h)
  inc = 0
  start = time.time()
  for c in range(C):
    for j in range(dH):
      for i in range(dW):
        x, y = i * h + 2 , j * h + 2

        x1 = 1 + x - math.floor(x)
        x2 = x - math.floor(x)
        x3 = math.floor(x) + 1 - x
        x4 = math.floor(x) + 2 - x

        y1 = 1 + y - math.floor(y)
        y2 = y - math.floor(y)
        y3 = math.floor(y) + 1 - y
        y4 = math.floor(y) + 2 - y

        mat_l = np.matrix([[cubic_interpolate(x1, a), cubic_interpolate(x2, a), cubic_interpolate(x3, a), cubic_interpolate(x4, a)]])
        mat_m = np.matrix(
          [
            [img[int(y - y1), int(x - x1), c], img[int(y - y2), int(x - x1), c], img[int(y + y3), int(x - x1), c], img[int(y + y4), int(x - x1), c]],
            [img[int(y - y1), int(x - x2), c], img[int(y - y2), int(x - x2), c], img[int(y + y3), int(x - x2), c], img[int(y + y4), int(x - x2), c]],
            [img[int(y - y1), int(x + x3), c], img[int(y - y2), int(x + x3), c], img[int(y + y3), int(x + x3), c], img[int(y + y4), int(x + x3), c]],
            [img[int(y - y1), int(x + x4), c], img[int(y - y2), int(x + x4), c], img[int(y + y3), int(x + x4), c], img[int(y + y4), int(x + x4), c]]
          ]
        )

        mat_r = np.matrix([[cubic_interpolate(y1, a)], [cubic_interpolate(y2, a)], [cubic_interpolate(y3, a)], [cubic_interpolate(y4, a)]])
        dst[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)

        # Print progress
        inc = inc + 1
        sys.stderr.write('\r\033[K' + get_progressbar_str(inc / (C * dH * dW), start, dst, base))
        sys.stderr.flush()
    # print('check')
    
  sys.stderr.write('\n')
  sys.stderr.flush()
  return dst

if __name__ == '__main__':
  
  inc = 0
  filename = askopenfilename()
  base = os.path.basename(filename)
  print('path =', filename)

  img = cv2.imread(filename)

  ratio = float(input('scale factor = '))
  a = -1/2

  dist = bicubic(img, ratio, a, base)
  print('Completed!')
  cv2.imwrite(f'bicubic_{base}', dist)