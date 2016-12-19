#my_data=[['slashdot','USA','yes',18,'None'],
#        ['google','France','yes',23,'Premium'],
#        ['digg','USA','yes',24,'Basic'],
#        ['kiwitobes','France','yes',23,'Basic'],
#        ['google','UK','no',21,'Premium'],
#        ['(direct)','New Zealand','no',12,'None'],
#        ['(direct)','UK','no',21,'Basic'],
#        ['google','USA','no',24,'Premium'],
#        ['slashdot','France','yes',19,'None'],
#        ['digg','USA','no',18,'None'],
#        ['google','UK','no',18,'None'],
#        ['kiwitobes','UK','no',19,'None'],
#        ['digg','New Zealand','yes',12,'Basic'],
#        ['slashdot','UK','no',21,'None'],
#        ['google','UK','yes',18,'Basic'],
#        ['kiwitobes','France','yes',19,'Basic']]
#1 4 action
#2 5 drama
#3 6 thriller
my_data=[['male','action',18,'1'],
        ['male','drama',23,'2'],
        ['male','thriller',24,'3'],
        ['female','action',23,'4'],
        ['female','drama',18,'5'],
        ['female','thriller',12,'6'],
        ['male','action',22,'1'],
        ['male','drama',24,'2'],
        ['female','action',19,'4'],
        ['male','thriller',18,'3'],
        ['female','thriller',18,'6'],
        ['female','drama',19,'5'],
        ['female','action',12,'4'],
        ['male','action',21,'1'],
        ['female','thriller',24,'6'],
        ['male','action',19,'1'],
        ['male', 'drama', 24, '2'],
        ['female', 'action', 19, '4'],
        ['male', 'thriller', 18, '3'],
        ['female', 'thriller', 36, '6'],
        ['male', 'drama', 36, '2'],
        ['female', 'action', 12, '4'],
        ['male', 'action', 25, '1'],
        ['female', 'thriller', 25, '6'],
        ['female', 'drama', 23, '5']]

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

set1,set2=divideset(my_data,3,20)
print entropy(set1)
print entropy(set2)
print entropy(my_data)

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

tree=buildtree(my_data)

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

print
print ("['male','action',12]")
print "masuk ke kelas : ",
print classify(['male','action',12],tree)