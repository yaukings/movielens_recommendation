############################################################
# panda
import pandas as pd
import pprint

print "lala"
# u.user
u_cols = ['user_id', 'age', 'sex']
users = pd.read_csv('user.csv', sep=',', names=u_cols, usecols=range(3),encoding='latin-1')

# u.data
r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('data.csv', sep=',', names=r_cols,usecols=range(3),encoding='latin-1')

# print users

m_cols = ['movie_id', 'movie_title', 'genre']
movies = pd.read_csv('item-1.csv', sep=',', names=m_cols, usecols=range(3),encoding='latin-1')

movie_ratings = pd.merge(movies, ratings)
lens = pd.merge(movie_ratings, users)

print lens.columns

list_data = []

print "find"
for x in lens.iterrows():
    y,data1 = x
    if data1[6] == 'M':
        if data1[2] == "unknown":
            r='1'
        if data1[2] == "Action":
            r='2'
        if data1[2] == "Adventure":
            r='3'
        if data1[2] == "Animation":
            r='4'
        if data1[2] == "Children's":
            r='5'
        if data1[2] == "Comedy":
            r='6'
        if data1[2] == "Crime":
            r='7'
        if data1[2] == "Documentary":
            r='8'
        if data1[2] == "Drama":
            r='9'
        if data1[2] == "Fantasy":
            r='10'
        if data1[2] == "Film-Noir":
            r='11'
        if data1[2] == "Horror":
            r='12'
        if data1[2] == "Musical":
            r='13'
        if data1[2] == "Mystery":
            r='14'
        if data1[2] == "Romance":
            r='15'
        if data1[2] == "Sci-Fi":
            r='16'
        if data1[2] == "Thriller":
            r='17'
        if data1[2] == "War":
            r='18'
        if data1[2] == "Western":
            r='19'
    elif data1[6]=='F':
        if data1[2] == "unknown":
            r='20'
        if data1[2] == "Action":
            r='21'
        if data1[2] == "Adventure":
            r='22'
        if data1[2] == "Animation":
            r='23'
        if data1[2] == "Children's":
            r='24'
        if data1[2] == "Comedy":
            r='25'
        if data1[2] == "Crime":
            r='26'
        if data1[2] == "Documentary":
            r='27'
        if data1[2] == "Drama":
            r='28'
        if data1[2] == "Fantasy":
            r='29'
        if data1[2] == "Film-Noir":
            r='30'
        if data1[2] == "Horror":
            r='31'
        if data1[2] == "Musical":
            r='32'
        if data1[2] == "Mystery":
            r='33'
        if data1[2] == "Romance":
            r='34'
        if data1[2] == "Sci-Fi":
            r='35'
        if data1[2] == "Thriller":
            r='36'
        if data1[2] == "War":
            r='37'
        if data1[2] == "Western":
            r='38'
        
    list_data.append((y,data1[1],data1[6],data1[5],data1[2],r))
    #print list_data

#pp = pprint.PrettyPrinter()

#pp.pprint(list_data)


        
############################################## start 22;

#==============================================================================
# for t in range(0,len(list_data)):
#     my_data[t][0]=list_data[t][2]
#     my_data[t][1]=list_data[t][4]
#     my_data[t][2]=list_data[t][3]
#     my_data[t][3]=list_data[t][5]
#==============================================================================

#print len(my_data)
# Divides a set on a specific column. Can handle numeric or nominal values
def divideset(rows, column, value):
    # Make a function that tells us if a row is in the first group (true) or the second group (false)
    split_function = None
    if isinstance(value, int) or isinstance(value, float):  # check if the value is a number i.e int or float
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value

    # Divide the rows into two sets and return them
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)

# print divideset(my_data,2,'yes')
# print divideset(my_data,3,20)

# Create counts of possible results (the last column of each row is the result)
def uniquecounts(rows):
   results={}
   for row in rows:
      # The result is the last column
      r=row[len(row)-1]
      if r not in results: results[r]=0
      results[r]+=1
   return results

# print(uniquecounts(my_data))

# Entropy is the sum of p(x)log(p(x)) across all
# the different possible results
def entropy(rows):
   from math import log
   log2=lambda x:log(x)/log(2)
   results=uniquecounts(rows)
   # Now calculate the entropy
   ent=0.0
   for r in results.keys():
      p=float(results[r])/len(rows)
      ent=ent-p*log2(p)
   return ent

set1,set2=divideset(list_data,5,1000)
print entropy(set1)
print entropy(set2)
print entropy(list_data)

class decisionnode:
  def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
    self.col=col
    self.value=value
    self.results=results
    self.tb=tb
    self.fb=fb

def buildtree(rows, scoref=entropy):  # rows is the set, either whole dataset or part of it in the recursive call,
  #  scoref is the method to measure heterogeneity. By default it's entropy.
  if len(rows) == 0: return decisionnode()  # len(rows) is the number of units in a set
  current_score = scoref(rows)

  # Set up some variables to track the best criteria
  best_gain = 0.0
  best_criteria = None
  best_sets = None

  column_count = len(rows[0]) - 1  # count the # of attributes/columns.
  # It's -1 because the last one is the target attribute and it does not count.
  for col in range(0, column_count):
      # Generate the list of all possible different values in the considered column
      global column_values  # Added for debugging
      column_values = {}
      for row in rows:
          column_values[row[col]] = 1
          # Now try dividing the rows up for each value in this column
      for value in column_values.keys():  # the 'values' here are the keys of the dictionnary
          (set1, set2) = divideset(rows, col, value)  # define set1 and set2 as the 2 children set of a division

          # Information gain
          p = float(len(set1)) / len(rows)  # p is the size of a child set relative to its parent
          gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)  # cf. formula information gain
          if gain > best_gain and len(set1) > 0 and len(set2) > 0:  # set must not be empty
              best_gain = gain
              best_criteria = (col, value)
              best_sets = (set1, set2)

  # Create the sub branches
  if best_gain > 0:
      trueBranch = buildtree(best_sets[0])
      falseBranch = buildtree(best_sets[1])
      return decisionnode(col=best_criteria[0], value=best_criteria[1],
                          tb=trueBranch, fb=falseBranch)
  else:
      return decisionnode(results=uniquecounts(rows))

tree=buildtree(list_data)

def printtree(tree, indent=''):
   # Is this a leaf node?
    if tree.results!=None:
        print(str(tree.results))
    else:
        print(str(tree.col)+':'+str(tree.value)+'? ')
        # Print the branches
        print indent+'T->',
        printtree(tree.tb,indent+'  ')
        print indent+'F->',
        printtree(tree.fb,indent+'  ')

printtree(tree)
def getwidth(tree):
    if tree.tb == None and tree.fb == None: return 1
    return getwidth(tree.tb) + getwidth(tree.fb)


def getdepth(tree):
    if tree.tb == None and tree.fb == None: return 0
    return max(getdepth(tree.tb), getdepth(tree.fb)) + 1


from PIL import Image, ImageDraw


def drawtree(tree, jpeg='tree.jpg'):
    w = getwidth(tree) * 100
    h = getdepth(tree) * 100 + 120

    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    drawnode(draw, tree, w / 2, 20)
    img.save(jpeg, 'JPEG')


def drawnode(draw, tree, x, y):
    if tree.results == None:
        # Get the width of each branch
        w1 = getwidth(tree.fb) * 100
        w2 = getwidth(tree.tb) * 100

        # Determine the total space required by this node
        left = x - (w1 + w2) / 2
        right = x + (w1 + w2) / 2

        # Draw the condition string
        draw.text((x - 20, y - 10), str(tree.col) + ':' + str(tree.value), (0, 0, 0))

        # Draw links to the branches
        draw.line((x, y, left + w1 / 2, y + 100), fill=(255, 0, 0))
        draw.line((x, y, right - w2 / 2, y + 100), fill=(255, 0, 0))

        # Draw the branch nodes
        drawnode(draw, tree.fb, left + w1 / 2, y + 100)
        drawnode(draw, tree.tb, right - w2 / 2, y + 100)
    else:
        txt = ' \n'.join(['%s:%d' % v for v in tree.results.items()])
        draw.text((x - 20, y), txt, (0, 0, 0))

drawtree(tree,jpeg='treeview.jpg')

def classify(observation,tree):
  if tree.results!=None:
    return tree.results
  else:
    v=observation[tree.col]
    branch=None
    if isinstance(v,int) or isinstance(v,float):
      if v>=tree.value: branch=tree.tb
      else: branch=tree.fb
    else:
      if v==tree.value: branch=tree.tb
      else: branch=tree.fb
    return classify(observation,branch)

kelas1=[]
kelas2=[]
kelas3=[]
kelas4=[]
kelas5=[]
kelas6=[]
kelas7=[]
kelas8=[]
kelas9=[]
kelas10=[]
kelas11=[]
kelas12=[]
kelas13=[]
kelas14=[]
kelas15=[]
kelas16=[]
kelas17=[]
kelas18=[]
kelas19=[]
kelas20=[]
kelas21=[]
kelas22=[]
kelas23=[]
kelas24=[]
kelas25=[]
kelas26=[]
kelas27=[]
kelas28=[]
kelas29=[]
kelas30=[]
kelas31=[]
kelas32=[]
kelas33=[]
kelas34=[]
kelas35=[]
kelas36=[]
kelas37=[]
kelas38=[]

for x in 1000:
    hasil3 = classify([list_data[x][0], list_data[x][1], list_data[x][2],list_data[3],list_data[4]], tree)
    hasil3 = str(hasil3)
    hasil3 = int(hasil3.split(':')[0][2:-1])
    if hasil3 == 1:
        kelas1.append(x)
    elif hasil3 == 2:
        kelas2.append(x)
    elif hasil3 == 3:
        kelas3.append(x)
    elif hasil3 == 4:
        kelas4.append(x)
    elif hasil3 == 5:
        kelas5.append(x)
    elif hasil3 == 6:
        kelas6.append(x)
    elif hasil3 == 7:
        kelas2.append(x)
    elif hasil3 == 8:
        kelas3.append(x)
    elif hasil3 == 9:
        kelas4.append(x)
    elif hasil3 == 10:
        kelas5.append(x)
    elif hasil3 == 11:
        kelas6.append(x)
    elif hasil3 == 12:
        kelas2.append(x)
    elif hasil3 == 13:
        kelas3.append(x)
    elif hasil3 == 14:
        kelas4.append(x)
    elif hasil3 == 15:
        kelas5.append(x)
    elif hasil3 == 16:
        kelas6.append(x)
    elif hasil3 == 17:
        kelas2.append(x)
    elif hasil3 == 18:
        kelas3.append(x)
    elif hasil3 == 19:
        kelas4.append(x)
    elif hasil3 == 20:
        kelas2.append(x)
    elif hasil3 == 21:
        kelas3.append(x)
    elif hasil3 == 22:
        kelas4.append(x)
    elif hasil3 == 23:
        kelas5.append(x)
    elif hasil3 == 24:
        kelas6.append(x)
    elif hasil3 == 25:
        kelas2.append(x)
    elif hasil3 == 26:
        kelas3.append(x)
    elif hasil3 == 27:
        kelas4.append(x)
    elif hasil3 == 28:
        kelas5.append(x)
    elif hasil3 == 29:
        kelas6.append(x)
    elif hasil3 == 30:
        kelas2.append(x)
    elif hasil3 == 31:
        kelas3.append(x)
    elif hasil3 == 32:
        kelas4.append(x)
    elif hasil3 == 33:
        kelas3.append(x)
    elif hasil3 == 34:
        kelas4.append(x)
    elif hasil3 == 35:
        kelas5.append(x)
    elif hasil3 == 36:
        kelas6.append(x)
    elif hasil3 == 37:
        kelas2.append(x)
    elif hasil3 == 38:
        kelas3.append(x)
    
print "Decision Tree Sukses"
#==============================================================================
# print kelas1
# 
# 
# 
# print
# print ("['male','action',12]")
# print "masuk ke kelas : ",
# 
# hasil = classify(['male','action',12],tree)
# hasil2 = str(hasil)
# 
# hasil2 = int(hasil2.split(':')[0][2:-1])
# print hasil2
#==============================================================================















#############################################################################
# Cluster K-Means

import sys
import math
import random
import subprocess

data=[  [4,18],
        [5,23],
        [3,24],
        [2,23],
        [1,18],
        [4,12],
        [2,22],
        [2,35],
        [1,19],
        [2,18],
        [3,18],
        [4,45],
        [5,12],
        [1,21],
        [1,24],
        [2,19],
        [3, 24],
        [3,19],
        [4,18],
        [5,36],
        [3,36],
        [1,12],
        [2,25],
        [2,50],
        [3,23]]

def main():
    # How many points are in our dataset?
    num_points = 25

    # For each of those points how many dimensions do they have?
    dimensions = 2

    # Bounds for the values of those points in each dimension
    lower = 0
    upper = 5

    # The K in k-means. How many clusters do we assume exist?
    num_clusters = 8

    # When do we say the optimization has 'converged' and stop updating clusters

    # if hasil3 == 1:
    #     data=kelas1
    # if hasil3 == 2:
    #     data=kelas2
    # if hasil3 == 3:
    #     data=kelas3
    # if hasil3 == 4:
    #     data=kelas4
    # if hasil3 == 5:
    #     data = kelas4
    # if hasil3 == 6:
    #     data = kelas4
    # if hasil3 == 7:
    #     data=kelas1
    # if hasil3 == 8:
    #     data=kelas2
    # if hasil3 == 9:
    #     data=kelas3
    # if hasil3 == 10:
    #     data=kelas4
    # if hasil3 == 11:
    #     data = kelas4
    # if hasil3 == 12:
    #     data = kelas4
    # if hasil3 == 13:
    #     data=kelas1
    # if hasil3 == 14:
    #     data=kelas2
    # if hasil3 == 15:
    #     data=kelas3
    # if hasil3 == 16:
    #     data=kelas4
    # if hasil3 == 17:
    #     data = kelas4
    # if hasil3 == 18:
    #     data = kelas4
    # if hasil3 == 19:
    #     data=kelas1
    # if hasil3 == 20:
    #     data=kelas2
    # if hasil3 == 21:
    #     data=kelas3
    # if hasil3 == 22:
    #     data=kelas4
    # if hasil3 == 23:
    #     data = kelas4
    # if hasil3 == 24:
    #     data = kelas4
    # if hasil3 == 25:
    #     data=kelas3
    # if hasil3 == 26:
    #     data=kelas4
    # if hasil3 == 27:
    #     data = kelas4
    # if hasil3 == 28:
    #     data = kelas4
    # if hasil3 == 29:
    #     data=kelas1
    # if hasil3 == 30:
    #     data=kelas2
    # if hasil3 == 31:
    #     data=kelas3
    # if hasil3 == 32:
    #     data=kelas4
    # if hasil3 == 33:
    #     data = kelas4
    # if hasil3 == 34:
    #     data = kelas4
    # if hasil3 == 35:
    #     data=kelas3
    # if hasil3 == 36:
    #     data=kelas4
    # if hasil3 == 37:
    #     data = kelas4
    # if hasil3 == 38:
    #     data = kelas4
    opt_cutoff = 0.5
    points = []
    # Generate some points
    for x in range(0, 25):
        points.append(Point(data[x]))

    #points.append(data[0])
    #print "haha"
    #print data[0]
    #points = [makeRandomPoint(dimensions, lower, upper)]
    #points = data
    # Cluster those data!
    clusters = kmeans(points, num_clusters, opt_cutoff)

    # Print our clusters
    for i, c in enumerate(clusters):
        for p in c.points:
            print " Cluster: ", i, "\t Point :", p

    # Display clusters using plotly for 2d data
    # This uses the 'open' command on a URL and may only work on OSX.
    #if dimensions == 2 and PLOTLY_USERNAME:
    #    print "Plotting points, launching browser ..."
    tew = 1
    if tew > 0 and tew <= 20:
        print "1"
    elif tew > 20 and tew <= 30:
        print "2"
    elif tew > 30 and tew <= 20:
        print "3"
    elif tew > 40:
        print "4"





class Point:
    '''
    An point in n dimensional space
    '''

    def __init__(self, coords):
        '''
        coords - A list of values, one per dimension
        '''

        self.coords = coords
        self.n = len(coords)

    def __repr__(self):

        return str(self.coords)

class Cluster:
    '''
    A set of points and their centroid
    '''

    def __init__(self, points):
        '''
        points - A list of point objects
        '''

        if len(points) == 0: raise Exception("ILLEGAL: empty cluster")
        # The points that belong to this cluster
        self.points = points

        # The dimensionality of the points in this cluster
        self.n = points[0].n


        # Assert that all points are of the same dimensionality
        for p in points:
            if p.n != self.n: raise Exception("ILLEGAL: wrong dimensions")

        # Set up the initial centroid (this is usually based off one point)
        self.centroid = self.calculateCentroid()

    def __repr__(self):
        '''
        String representation of this object
        '''
        return str(self.points)

    def update(self, points):
        '''
        Returns the distance between the previous centroid and the new after
        recalculating and storing the new centroid.
        '''
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        shift = getDistance(old_centroid, self.centroid)
        return shift

    def calculateCentroid(self):
        '''
        Finds a virtual center point for a group of n-dimensional points
        '''
        numPoints = len(self.points)
        # Get a list of all coordinates in this cluster
        coords = [p.coords for p in self.points]
        # Reformat that so all x's are together, all y'z etc.
        unzipped = zip(*coords)
        # Calculate the mean for each dimension
        centroid_coords = [math.fsum(dList) / numPoints for dList in unzipped]

        return Point(centroid_coords)


def kmeans(points, k, cutoff):
    # Pick out k random points to use as our initial centroids
    initial = []
    initial.append(Point([4,12]))
    initial.append(Point([5, 23]))
    initial.append(Point([5, 36]))
    initial.append(Point([4, 45]))
    initial.append(Point([2, 18]))
    initial.append(Point([2, 23]))
    initial.append(Point([2, 35]))
    initial.append(Point([2, 50]))

    #initial = random.sample(points, k)
    print "initial: ",
    print initial
    # Create k clusters using those centroids
    clusters = [Cluster([p]) for p in initial]


    # Loop through the dataset until the clusters stabilize
    loopCounter = 0
    while True:
        # Create a list of lists to hold the points in each cluster
        lists = [[] for c in clusters]
        clusterCount = len(clusters)

        # Start counting loops
        loopCounter += 1
        # For every point in the dataset ...
        for p in points:
            # Get the distance between that point and the centroid of the first
            # cluster.
            smallest_distance = getDistance(p, clusters[0].centroid)

            # Set the cluster this point belongs to
            clusterIndex = 0

            # For the remainder of the clusters ...
            for i in range(clusterCount - 1):
                # calculate the distance of that point to each other cluster's
                # centroid.
                distance = getDistance(p, clusters[i + 1].centroid)
                # If it's closer to that cluster's centroid update what we
                # think the smallest distance is, and set the point to belong
                # to that cluster
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i + 1
            lists[clusterIndex].append(p)

        # Set our biggest_shift to zero for this iteration
        biggest_shift = 0.0

        # As many times as there are clusters ...
        for i in range(clusterCount):
            # Calculate how far the centroid moved in this iteration
            shift = clusters[i].update(lists[i])
            # Keep track of the largest move from all cluster centroid updates
            biggest_shift = max(biggest_shift, shift)

        # If the centroids have stopped moving much, say we're done!
        if biggest_shift < cutoff:
            print "Converged after %s iterations" % loopCounter
            break
    return clusters


def getDistance(a, b):
    '''
    Euclidean distance between two n-dimensional points.
    Note: This can be very slow and does not scale well
    '''
    if a.n != b.n:
        raise Exception("ILLEGAL: non comparable points")

    ret = reduce(lambda x, y: x + pow((a.coords[y] - b.coords[y]), 2), range(a.n), 0.0)
    return math.sqrt(ret)


def makeRandomPoint(n, lower, upper):
    '''
    Returns a Point object with n dimensions and values between lower and
    upper in each of those dimensions
    '''
    p = Point([random.uniform(lower, upper) for i in range(2)])
    print p
    #print (p)

    return p

def recommendation():
    print "Recommended for you :"
    # for x in range(0, 3):
    #     print "movie_name"

if __name__ == "__main__":
    main()
    #main(hasil3)
    recommendation()
    #recommendation(hasil3)




