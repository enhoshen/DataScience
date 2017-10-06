import sys
import ssl
import matplotlib.pyplot as plt
import urllib
dataUrl = 'https://ceiba.ntu.edu.tw/course/481ea4/hw1_data.csv'

def urlDownload ( url ):
    context = ssl._create_unverified_context()
    return  urllib.request.urlopen( url, context=context )
def parseBytes ( line ) :
    line = line.decode('utf-8').split('\n')[0] 
    return [x for x in line.split(',')]
def parseTable  ( csvfile ):
    attrbList = {}
    pop2percDict = {}
    classDict = {}
    currentClass = ''
    for bLine in csvfile : 
        line = parseBytes(bLine)
        if line[0] == 'class':
            for i in range(len(line)-1):
                attrbN = line[i+1] # attribute name
                attrbList[attrbN] = i 
                attrbN = attrbN.split(' ')
                if attrbN[-1] == 'population':
                    pop2percDict [attrbN[0] ] = i
                if attrbN[-1] == 'percentage':
                    pop2percDict [attrbN[0] ] = (pop2percDict[attrbN[0]],i)

        elif line[1] == '':
            classDict[line[0][0]]= (line[0],[],[])
            currentClass= line[0][0]
        else :
            c = line[0]
            attrbs = [float(i) for i in line[1:] ]
            classDict[currentClass][1].append (c)
            popPercIdx = [list(i) for i in  zip( *pop2percDict.values()) ] 
            totalpop = sum([attrbs[i] for i in popPercIdx[0] ] ) 
            totalsmokepop = sum( [attrbs[v[0]]*attrbs[v[1]]  for _ , v in pop2percDict.items() ]  )/100 
            totalperc = totalsmokepop/totalpop*100 
            classDict[currentClass][2].append ( attrbs+  [totalpop , totalsmokepop , totalperc] )   
    attrbList['totalpop']=len(line)-1
    attrbList['totalsmokepop']=len(line)
    attrbList['totalperc']=len(line)+1
    pop2percDict ['total']=(len(line)-1, len(line)+1)
    return classDict , attrbList, pop2percDict

def plotting ( arg , classDict , attrbList , pop2percDict):
    fig , ax = plt.subplots(figsize=(10,10*9/16))
    cls, subclass ,table  = classDict[arg[0]]
    chartT = arg[1]
    tableSize = len( subclass )
    popNum = len( pop2percDict.keys())
    plt.title( "Smoking percentage vs "+cls  )   
    handles = []
    labelList=[]
    if chartT == 'b' or chartT == 'l':
        for i , key in enumerate(pop2percDict):
            perc = pop2percDict[key][1]
            percList = [  row[perc]  for row in table]
            for x , y in zip([(a-0.4) for a in range (tableSize)], percList):
                if chartT == 'b':
                    ax.text ( x+0.1+i*0.2,y+1, '%.2f' % y )
                else :
                    ax.text ( x+0.3 , y+1 , '%.2f' % y )
            if chartT == 'b':
                handles.append(  ax.bar( left=[ (a +i*0.2 -0.2) for a in range(tableSize) ], height= percList, width = 0.2 ) )
            else :
                handles.append(  ax.plot( [a for a in range(tableSize) ] , percList, '.-' )[0] )
            labelList.append(key)
        plt.xticks( range(tableSize) , [key for key  in subclass] )
        plt.ylabel( "Smoking percentage (%) ")
        plt.xlabel( cls )
        plt.legend(handles,labelList)
    
    if chartT == 'p':
        totalClsPop = sum( [ row[attrbList['totalpop'] ] for row in table])
        percList = [ row[attrbList['totalsmokepop']]/totalClsPop*100 for row in table ]
        labelList = [ key for  key  in subclass] 
        ax.pie ( percList , labels=labelList , autopct = '%1.1f%%'  )
        plt.subplots_adjust(left=0.27,right =0.74,top=0.92)

    plt.show()
if __name__ == "__main__" :
    args = [sys.argv[i+1].split('-')[1] for i in range(len(sys.argv)-1) ]   
    smokePopData = urlDownload ( dataUrl )
    dataTable = smokePopData.readlines()
    table,attrb,pop2perc = parseTable(dataTable)
    plt.rcdefaults()    
    for arg in args:
        plotting ( arg, table, attrb, pop2perc)

